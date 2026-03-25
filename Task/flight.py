'''
Minimal flight flow:
1) take off
2) wait for camera feed
3) detect square
4) center square
5) check yellow junction lines
6) land
'''
import os
# MUST set before importing cv2 so Qt picks up xcb backend
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import cv2
import numpy as np
from pymavlink import mavutil
from control import Control
import time
from sensor import Camera
import queue

# Percentage of strip resolution kept for sensor processing (1-100).
STRIP_DOWNSAMPLE_PERCENT = 20

# Line-follow tuning globals.
LINE_FOLLOW_DURATION_SECONDS = 22
LINE_FOLLOW_PID_KP = 190
LINE_FOLLOW_PID_KI = 0.002
LINE_FOLLOW_PID_KD = 200
# 0% => only turning, 100% => only left/right movement.
LINE_FOLLOW_TURN_ADJUST_PERCENT = 5

class Brain:
    def __init__(self):
        self.control = Control()
        self.camera = Camera()
        self._latest_frame = None
        self._latest_display = None
        self._frame_queue = queue.Queue(maxsize=2)  # main-thread display queue
        self._last_center_cmd_time = 0.0
        self._vertical_cmd_sign = -1.0  # start inverted: prior mapping increased top/bottom error
        self._last_err_y_abs = None
        self._last_cmd_axis = None
        self._err_x_filt = 0.0
        self._err_y_filt = 0.0
        self._last_cmd_sign_x = 0
        self._last_cmd_sign_y = 0
        self._settle_until = 0.0
        self._line_pid_kp = LINE_FOLLOW_PID_KP
        self._line_pid_ki = LINE_FOLLOW_PID_KI
        self._line_pid_kd = LINE_FOLLOW_PID_KD
        self._line_pid_integral = 0.0
        self._line_pid_prev_error = 0.0
        self._line_pid_prev_time = None
        self._last_yaw_cmd_time = 0.0

        # Legacy logic intentionally removed:
        # - PID line following
        # - AprilTag detection
        # - pad centering and advanced landing

    def process_frame(self, frame):
        """Cache latest color frame and build annotated preview."""
        self._latest_frame = frame
        display = frame.copy()
        cv2.rectangle(display, (0, 0), (250, 28), (0, 0, 0), -1)
        cv2.putText(display, "Color Camera Feed", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        self._latest_display = display

    def detect_square_and_print_each_frame(self, frame):
        """Detect a gray square box and print detection status for every frame."""
        annotated = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # Gray in HSV has low saturation. Keep broad value range for lighting changes.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_mask = cv2.inRange(hsv, (0, 0, 40), (179, 55, 220))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_square = None
        best_score = float('-inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200:
                continue
            if area > frame_area * 0.35:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h if h else 1)
            if not (0.80 <= aspect <= 1.25):
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            dist_to_center = abs(cx - frame_w / 2.0) + abs(cy - frame_h / 2.0)
            # Prefer larger squares close to image center.
            score = area - 6.0 * dist_to_center
            if score > best_score:
                best_score = score
                best_square = (x, y, w, h)

        if best_square is not None:
            x, y, w, h = best_square
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Gray square detected", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("[VISION] Gray square detected")
        else:
            cv2.putText(annotated, "Gray square not detected", (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print("[VISION] Gray square not detected")

        return annotated, best_square

    def center_to_box(self, frame, box):
        """Center detected box in the frame using horizontal then vertical corrections."""
        if box is None:
            print("[CENTER] No box, skip centering")
            return frame, False

        x, y, w, h = box
        frame_h, frame_w = frame.shape[:2]

        left_len = x
        right_len = frame_w - (x + w)
        top_len = y
        bottom_len = frame_h - (y + h)

        # Draw green guide lines from the box to frame edges.
        cy = y + h // 2
        cx = x + w // 2
        cv2.line(frame, (0, cy), (x, cy), (0, 255, 0), 2)                   # left
        cv2.line(frame, (x + w, cy), (frame_w - 1, cy), (0, 255, 0), 2)      # right
        cv2.line(frame, (cx, 0), (cx, y), (0, 255, 0), 2)                    # top
        cv2.line(frame, (cx, y + h), (cx, frame_h - 1), (0, 255, 0), 2)      # bottom

        err_x = left_len - right_len
        err_y = top_len - bottom_len
        # Low-pass filter helps avoid over-correction when camera/control feedback is delayed.
        alpha = 0.75
        self._err_x_filt = alpha * self._err_x_filt + (1.0 - alpha) * err_x
        self._err_y_filt = alpha * self._err_y_filt + (1.0 - alpha) * err_y

        cv2.putText(frame, f"L:{left_len} R:{right_len}", (8, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"T:{top_len} B:{bottom_len}", (8, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"CX:{cx} CY:{cy}", (8, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"FX:{self._err_x_filt:+.1f} FY:{self._err_y_filt:+.1f}", (8, 138),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        print(f"[CENTER] L={left_len} R={right_len} | T={top_len} B={bottom_len} | CX={cx} CY={cy} | err_x={err_x} err_y={err_y} | filt_x={self._err_x_filt:+.1f} filt_y={self._err_y_filt:+.1f}")

        # Horizontal first, then vertical, with hysteresis to suppress oscillation.
        tol_x_high = 26
        tol_x_low = 14 * 1.10
        tol_y_high = 26
        tol_y_low = 14 * 1.10
        ex = abs(self._err_x_filt)
        ey = abs(self._err_y_filt)
        need_x = ex > tol_x_high or (self._last_cmd_axis == "x" and ex > tol_x_low)
        need_y = ey > tol_y_high or (self._last_cmd_axis == "y" and ey > tol_y_low)

        now = time.time()
        if now < self._settle_until:
            print(f"[CENTER] Settling... wait {self._settle_until - now:.2f}s")
            return frame, False

        if now - self._last_center_cmd_time < 0.25:
            return frame, False

        if need_x:
            # err_x > 0 means box is to the right of center, move right (+vy).
            sign_x = 1 if self._err_x_filt > 0 else -1
            if self._last_cmd_sign_x != 0 and sign_x != self._last_cmd_sign_x and ex < (tol_x_high + 10):
                print("[CENTER] X sign flip near center, skipping one cycle")
                self._settle_until = now + 0.35
                return frame, False

            gain_x = min(1.0, max(0.30, ex / 140.0))
            vy = sign_x * (0.0155792 * gain_x)
            self.control.move_with_velocity(vx=0.0, vy=vy, vz=0.0, duration=0.08)
            print(f"[CENTER] Horizontal adjust vy={vy:+.2f}")
            self._last_center_cmd_time = now
            self._last_cmd_axis = "x"
            self._last_cmd_sign_x = sign_x
            self._settle_until = now + 0.45
        elif need_y:
            # Auto-correct vertical mapping if error trend gets worse.
            err_y_abs = ey
            if self._last_cmd_axis == "y" and self._last_err_y_abs is not None and err_y_abs > (self._last_err_y_abs + 3):
                self._vertical_cmd_sign *= -1.0
                print(f"[CENTER] Vertical mapping flipped. New sign={self._vertical_cmd_sign:+.0f}")

            # err_y > 0 means box lower than center. Sign adapts if mapping is inverted.
            sign_y = 1 if self._err_y_filt > 0 else -1
            if self._last_cmd_sign_y != 0 and sign_y != self._last_cmd_sign_y and ey < (tol_y_high + 10):
                print("[CENTER] Y sign flip near center, skipping one cycle")
                self._settle_until = now + 0.35
                return frame, False

            gain_y = min(1.0, max(0.30, ey / 140.0))
            base_vx = sign_y * (0.0155792 * gain_y)
            vx = self._vertical_cmd_sign * base_vx
            self.control.move_with_velocity(vx=vx, vy=0.0, vz=0.0, duration=0.08)
            print(f"[CENTER] Vertical adjust vx={vx:+.2f}")
            self._last_center_cmd_time = now
            self._last_cmd_axis = "y"
            self._last_err_y_abs = err_y_abs
            self._last_cmd_sign_y = sign_y
            self._settle_until = now + 0.45
        else:
            print("[CENTER] Box centered within tolerance")
            self._last_cmd_axis = None
            self._last_err_y_abs = ey
            return frame, True

        return frame, False

    def junction_logic(self, frame, box):
        """For post-centering check: report whether yellow exists and where it appears."""
        if box is None:
            print("[YELLOW] No square, skip yellow check")
            return frame, None

        frame_h, frame_w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Keep the existing yellow range; morphology below improves robustness.
        yellow_mask = cv2.inRange(hsv, (18, 80, 80), (40, 255, 255))

        # Remove speckles and connect slightly broken line segments.
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, small_kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ignore small yellow noise; merge significant contours for one location estimate.
        min_area = 140
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        if valid:
            all_pts = cv2.vconcat(valid)
            M = cv2.moments(all_pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(all_pts)
                cx = x + w // 2
                cy = y + h // 2

            frame_cx = frame_w // 2
            frame_cy = frame_h // 2
            tol_x = 25
            tol_y = 25

            if cx < frame_cx - tol_x:
                horiz = "left"
            elif cx > frame_cx + tol_x:
                horiz = "right"
            else:
                horiz = "center"

            if cy < frame_cy - tol_y:
                vert = "top"
            elif cy > frame_cy + tol_y:
                vert = "bottom"
            else:
                vert = "middle"

            location = f"{vert}-{horiz}"
            print(f"[YELLOW] Detected: yes | where: {location}")

            cv2.drawContours(frame, valid, -1, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.line(frame, (frame_cx, 0), (frame_cx, frame_h - 1), (255, 255, 0), 1)
            cv2.line(frame, (0, frame_cy), (frame_w - 1, frame_cy), (255, 255, 0), 1)
            cv2.putText(frame, f"Yellow: yes ({location})", (8, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            return frame, location

        print("[YELLOW] Detected: no")
        cv2.putText(frame, "Yellow: no", (8, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return frame, None

    def _get_top_strip(self, frame):
        """Keep top 20% of frame and remove bottom 80%."""
        strip_h = max(1, int(frame.shape[0] * 0.20))
        return frame[:strip_h, :].copy()

    def _build_yellow_sensor_array(self, strip):
        """Downsample strip, average vertically, and return 1D yellow binary array."""
        scale = max(1, min(100, int(STRIP_DOWNSAMPLE_PERCENT))) / 100.0
        new_w = max(8, int(strip.shape[1] * scale))
        new_h = max(4, int(strip.shape[0] * scale))
        small = cv2.resize(strip, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Average over vertical axis -> one row of BGR pixels (sensor strip).
        avg_row_bgr = np.mean(small, axis=0, dtype=np.float32).astype(np.uint8)
        avg_row_bgr = avg_row_bgr.reshape((1, new_w, 3))

        hsv_row = cv2.cvtColor(avg_row_bgr, cv2.COLOR_BGR2HSV)
        yellow_row = cv2.inRange(hsv_row, (18, 80, 80), (40, 255, 255))
        sensor = (yellow_row[0] > 0).astype(np.uint8)

        return sensor, small

    def _line_follow_from_strip(self, strip):
        """PID line follow using a mix of yaw turning and lateral movement."""
        # Keep only PID and mix ratio as global tunables; use fixed local drive constants.
        command_duration = 0.08
        error_adjust_speed_gain = 90
        forward_speed = 0.18
        lateral_max_speed = 0.025 * error_adjust_speed_gain
        yaw_rate_dps = 30.0 * error_adjust_speed_gain
        yaw_scale_deg = 90.0
        yaw_max_deg = 4.0 * error_adjust_speed_gain
        yaw_min_deg = 1.0
        yaw_deadband = 0.01
        yaw_command_interval = 0.15
        steer_gain = 100.0
        steer_limit = 0.16

        sensor, small = self._build_yellow_sensor_array(strip)
        w = sensor.shape[0]
        center = (w - 1) / 2.0

        yellow_idx = np.where(sensor == 1)[0]
        if yellow_idx.size == 0:
            # No yellow seen: drift forward slowly without lateral correction.
            self.control.move_with_velocity(vx=0.02, vy=0.0, vz=0.0, duration=command_duration)
            print(f"[LINE_FOLLOW] sensor={''.join(sensor.astype(str))} | no yellow")
            return strip

        line_center = float(np.mean(yellow_idx))
        error = (line_center - center) / max(center, 1.0)

        now = time.time()
        if self._line_pid_prev_time is None:
            dt = command_duration
        else:
            dt = max(0.03, now - self._line_pid_prev_time)

        self._line_pid_integral += error * dt
        self._line_pid_integral = float(np.clip(self._line_pid_integral, -1.5, 1.5))
        derivative = (error - self._line_pid_prev_error) / dt

        raw_steer = (
            self._line_pid_kp * error
            + self._line_pid_ki * self._line_pid_integral
            + self._line_pid_kd * derivative
        )
        scaled_steer = raw_steer * steer_gain
        steer = float(np.clip(scaled_steer, -steer_limit, steer_limit))
        steer_clipped = abs(scaled_steer) > steer_limit

        # Split correction: 0% means yaw-only, 100% means lateral-only.
        lateral_ratio = float(np.clip(LINE_FOLLOW_TURN_ADJUST_PERCENT / 100.0, 0.0, 1.0))
        yaw_ratio = 1.0 - lateral_ratio

        yaw_steer = steer * yaw_ratio
        lateral_steer = steer * lateral_ratio

        # Keep moving forward while applying lateral correction in the same velocity command.
        vy_unclipped = (lateral_steer / steer_limit) * lateral_max_speed
        vy = float(np.clip(
            vy_unclipped,
            -lateral_max_speed,
            lateral_max_speed,
        ))
        vy_clipped = abs(vy_unclipped) > lateral_max_speed
        vx = forward_speed
        self.control.move_with_velocity(vx=vx, vy=vy, vz=0.0, duration=command_duration)

        yaw_cmd_deg = 0.0
        yaw_unclipped = 0.0
        yaw_clipped = False
        if abs(yaw_steer) > yaw_deadband:
            yaw_unclipped = yaw_steer * yaw_scale_deg
            yaw_cmd_deg = float(np.clip(
                yaw_unclipped,
                -yaw_max_deg,
                yaw_max_deg,
            ))
            yaw_clipped = abs(yaw_unclipped) > yaw_max_deg
            if 0 < abs(yaw_cmd_deg) < yaw_min_deg:
                yaw_cmd_deg = yaw_min_deg if yaw_cmd_deg > 0 else -yaw_min_deg

            if now - self._last_yaw_cmd_time >= yaw_command_interval:
                self.control.master.mav.command_long_send(
                    self.control.master.target_system,
                    self.control.master.target_component,
                    mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                    0,
                    abs(yaw_cmd_deg),              # angle in degrees
                    yaw_rate_dps,                  # yaw speed deg/s
                    1 if yaw_cmd_deg > 0 else -1, # clockwise vs counter-clockwise
                    1,                             # relative
                    0, 0, 0
                )
                self._last_yaw_cmd_time = now

        self._line_pid_prev_error = error
        self._line_pid_prev_time = now

        print(
            f"[LINE_FOLLOW] sensor={''.join(sensor.astype(str))} | "
            f"line_center={line_center:.1f} error={error:+.3f} "
            f"raw={raw_steer:+.4f} scaled={scaled_steer:+.3f} steer={steer:+.3f} clipS={int(steer_clipped)} "
            f"mix={LINE_FOLLOW_TURN_ADJUST_PERCENT:.0f}%LR/{(100.0 - LINE_FOLLOW_TURN_ADJUST_PERCENT):.0f}%Yaw "
            f"vy={vy:+.3f} clipV={int(vy_clipped)} yawRaw={yaw_unclipped:+.2f} yaw={yaw_cmd_deg:+.2f} clipY={int(yaw_clipped)}"
        )

        preview = cv2.resize(small, (strip.shape[1], strip.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.putText(preview, "Line follow strip", (8, min(strip.shape[0] - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return preview


    def _push_display(self, disp):
        """Push annotated frame to the live preview window (main thread only)."""
        try:
            self._frame_queue.put_nowait(disp)
        except queue.Full:
            pass
        try:
            cv2.imshow('Drone Camera', self._frame_queue.get_nowait())
        except queue.Empty:
            pass
        cv2.waitKey(1)

    def start(self):
        """Take off, confirm camera feed, hold for 10 seconds, then land."""
        print("MAVLink connected. Starting simplified flight sequence...")

        # 1. Set GUIDED mode
        self.control.set_mode('GUIDED')

        # 2. Force arm
        self.control.force_arm()

        # 3. Takeoff
        target_alt_m = 1.8
        self.control.takeoff(target_alt_m)

        # 4. Start camera and wait for first frame
        cv2.namedWindow('Drone Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Camera', 640, 480)
        self.camera.start_thread(self.process_frame)
        print("Camera started. Waiting for first frame...")
        while self._latest_frame is None:
            time.sleep(0.05)

        print("Color camera feed received. Starting sequence: detect -> center -> yellow-check -> land")

        # 5. Run sequence phases
        phase = "wait_square"
        centered_since = None
        yellow_started_at = None
        line_follow_started_at = None
        center_hold_seconds = 1.2
        yellow_check_seconds = 2.0
        line_follow_seconds = LINE_FOLLOW_DURATION_SECONDS

        while True:
            frame = self._latest_frame
            if frame is not None:
                analysis, detected_box = self.detect_square_and_print_each_frame(frame.copy())

                if phase == "wait_square":
                    if detected_box is not None:
                        print("[PHASE] Square found -> centering phase")
                        phase = "center"
                    else:
                        print("[PHASE] Waiting for square...")

                elif phase == "center":
                    analysis, centered = self.center_to_box(analysis, detected_box)
                    if detected_box is None:
                        centered_since = None
                        phase = "wait_square"
                        print("[PHASE] Lost square -> back to waiting")
                    elif centered:
                        if centered_since is None:
                            centered_since = time.time()
                        if time.time() - centered_since >= center_hold_seconds:
                            phase = "junction"
                            yellow_started_at = time.time()
                            print("[PHASE] Center stable -> yellow check phase")
                    else:
                        centered_since = None

                elif phase == "junction":
                    analysis, yellow_location = self.junction_logic(analysis, detected_box)

                    if yellow_location == "top-center":
                        phase = "line_follow"
                        line_follow_started_at = time.time()
                        self._line_pid_integral = 0.0
                        self._line_pid_prev_error = 0.0
                        self._line_pid_prev_time = None
                        print("[PHASE] top-center detected -> line follow for 4.0s")

                    elapsed = time.time() - yellow_started_at if yellow_started_at else 0.0
                    print(f"[PHASE] Yellow check: {elapsed:.1f}/{yellow_check_seconds:.1f}s")
                    if elapsed >= yellow_check_seconds:
                        print("[PHASE] Yellow check complete -> landing")
                        break

                elif phase == "line_follow":
                    src = self._latest_frame if self._latest_frame is not None else frame
                    strip = self._get_top_strip(src)
                    analysis = self._line_follow_from_strip(strip)

                    elapsed = time.time() - line_follow_started_at if line_follow_started_at else 0.0
                    print(f"[PHASE] Line follow: {elapsed:.1f}/{line_follow_seconds:.1f}s")
                    if elapsed >= line_follow_seconds:
                        print("[PHASE] Line follow complete -> landing")
                        break

                if phase != "line_follow":
                    cv2.rectangle(analysis, (0, 0), (250, 28), (0, 0, 0), -1)
                    cv2.putText(analysis, "Color Camera Feed", (6, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                self._push_display(analysis)
            time.sleep(0.03)

        # 6. Land
        self.control.land()
        print("Landing command sent. Flight sequence complete.")
        cv2.destroyAllWindows()

    def __del__(self):
        """Destructor to ensure threads are stopped."""
        self.camera.stop_thread()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
    finally:
        del brain