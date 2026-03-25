'''
Minimal flight flow:
1) take off
2) wait for camera feed
3) keep full-color feed for 10 seconds
4) land
'''
import os
# MUST set before importing cv2 so Qt picks up xcb backend
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import cv2
from control import Control
import time
from sensor import Camera
import queue

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
            return frame

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
        tol_x_low = 14
        tol_y_high = 26
        tol_y_low = 14
        ex = abs(self._err_x_filt)
        ey = abs(self._err_y_filt)
        need_x = ex > tol_x_high or (self._last_cmd_axis == "x" and ex > tol_x_low)
        need_y = ey > tol_y_high or (self._last_cmd_axis == "y" and ey > tol_y_low)

        now = time.time()
        if now < self._settle_until:
            print(f"[CENTER] Settling... wait {self._settle_until - now:.2f}s")
            return frame

        if now - self._last_center_cmd_time < 0.25:
            return frame

        if need_x:
            # err_x > 0 means box is to the right of center, move right (+vy).
            sign_x = 1 if self._err_x_filt > 0 else -1
            if self._last_cmd_sign_x != 0 and sign_x != self._last_cmd_sign_x and ex < (tol_x_high + 10):
                print("[CENTER] X sign flip near center, skipping one cycle")
                self._settle_until = now + 0.35
                return frame

            gain_x = min(1.0, max(0.30, ex / 140.0))
            vy = sign_x * (0.01 * gain_x)
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
                return frame

            gain_y = min(1.0, max(0.30, ey / 140.0))
            base_vx = sign_y * (0.01 * gain_y)
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

        return frame

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
        target_alt_m = 1.3
        self.control.takeoff(target_alt_m)

        # 4. Start camera and wait for first frame
        cv2.namedWindow('Drone Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Camera', 640, 480)
        self.camera.start_thread(self.process_frame)
        print("Camera started. Waiting for first frame...")
        while self._latest_frame is None:
            time.sleep(0.05)

        print("Color camera feed received. Landing in 10 seconds...")

        # 5. Keep preview active for 10 seconds before landing
        land_at = time.time() + 10.0
        while time.time() < land_at:
            frame = self._latest_frame
            if frame is not None:
                analysis, detected_box = self.detect_square_and_print_each_frame(frame.copy())
                analysis = self.center_to_box(analysis, detected_box)
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