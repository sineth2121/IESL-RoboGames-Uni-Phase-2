'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import os
# MUST set before importing cv2 so Qt picks up xcb backend
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import cv2
import numpy as np
from control import Control
import time
from sensor import Camera
import queue

# ── HSV color ranges for detection (H: 0-179, S: 0-255, V: 0-255) ──────────
COLOR_RANGES = {
    'red':    ([  0,  80, 80], [ 10, 255, 255],   # lower red hue
               [160,  80, 80], [179, 255, 255]),   # upper red hue (wraps)
    'blue':   ([ 90, 80, 80], [130, 255, 255], None, None),
    'green':  ([ 35, 60, 60], [ 85, 255, 255], None, None),
    'yellow': ([ 20, 80, 80], [ 35, 255, 255], None, None),
}

# BGR colors for drawing bounding boxes
DRAW_COLOR = {
    'red':    (0,   0,   255),
    'blue':   (255, 0,   0),
    'green':  (0,   255, 0),
    'yellow': (0,   200, 255),
}

MIN_CONTOUR_AREA = 500   # px² — ignore tiny blobs


class Brain:
    def __init__(self):
        self.control = Control()
        self.camera = Camera()
        self._latest_detections = []
        self._latest_frame = None            # raw BGR frame for line follow
        self._latest_display = None          # annotated frame from process_frame
        self._frame_queue = queue.Queue(maxsize=2)  # main-thread display queue

        # PID state
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = None
        self._vy_smooth = 0.0          # smoothed lateral output
        self._deriv_smooth = 0.0       # EMA-filtered derivative
        self._prev_vy = 0.0            # for rate limiter

        # AprilTag detector (tag36h11 family used in world)
        self._tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self._tag_params = cv2.aruco.DetectorParameters()
        self._tag_detector = cv2.aruco.ArucoDetector(self._tag_dict, self._tag_params)
        self._tag_land_area = 20000  # px² — trigger landing only when drone is directly over pad

    # ── frame processing (runs in camera thread) ─────────────────────────────
    def process_frame(self, frame):
        """
        Detect colored objects in the frame, draw enhanced bounding boxes,
        and store detections.
        """
        self._latest_frame = frame          # store raw frame for line_follow()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display = frame.copy()
        overlay = frame.copy()
        detections = []
        fh, fw = display.shape[:2]

        for color_name, ranges in COLOR_RANGES.items():
            lo1, hi1, lo2, hi2 = ranges

            mask = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
            if lo2 is not None:
                mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                detections.append({
                    'color': color_name,
                    'bbox':  (x, y, w, h),
                    'center': (cx, cy),
                    'area':  area,
                })

                bgr = DRAW_COLOR[color_name]

                # Semi-transparent filled contour
                cv2.fillPoly(overlay, [cnt], bgr)

                # Corner-bracket style box
                t = max(2, min(w, h) // 6)   # bracket arm length
                lw = 2
                for px, py in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                    sx = 1 if px == x else -1
                    sy = 1 if py == y else -1
                    cv2.line(display, (px, py), (px + sx*t, py), bgr, lw+1)
                    cv2.line(display, (px, py), (px, py + sy*t), bgr, lw+1)
                # Thin full rect
                cv2.rectangle(display, (x, y), (x+w, y+h), bgr, 1)

                # Label with dark background
                label = f"{color_name}  {area:.0f}px"
                (lw2, lh2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x, y - lh2 - 10), (x + lw2 + 6, y), (0, 0, 0), -1)
                cv2.putText(display, label, (x + 3, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

                # Centre dot + crosshair
                cv2.circle(display, (cx, cy), 5, bgr, -1)
                cv2.circle(display, (cx, cy), 10, bgr, 1)

        # Blend semi-transparent overlay
        cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)

        # Frame-centre crosshair
        cv2.line(display, (fw//2 - 20, fh//2), (fw//2 + 20, fh//2), (220, 220, 220), 1)
        cv2.line(display, (fw//2, fh//2 - 20), (fw//2, fh//2 + 20), (220, 220, 220), 1)
        cv2.circle(display, (fw//2, fh//2), 4, (220, 220, 220), 1)

        # Top-left HUD
        cv2.rectangle(display, (0, 0), (200, 26), (0, 0, 0), -1)
        cv2.putText(display, f"Objects: {len(detections)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        self._latest_detections = detections
        self._latest_display = display      # save annotated frame for line_follow

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_detections(self):
        """Return the most recently detected objects (thread-safe read)."""
        return list(self._latest_detections)

    def _detect_apriltag(self, frame):
        """Detect AprilTag in frame. Returns (tag_id, cx, cy, area, corners) or None."""
        corners, ids, _ = self._tag_detector.detectMarkers(frame)
        if ids is None or len(ids) == 0:
            return None
        best, best_area = None, 0
        for i, tc in enumerate(corners):
            c = tc[0]
            w = np.linalg.norm(c[0] - c[1])
            h = np.linalg.norm(c[0] - c[3])
            area = w * h
            if area > best_area:
                best_area = area
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                best = (int(ids[i][0]), cx, cy, area, tc)
        return best

    def _pid(self, error, kp=0.0004, ki=0.0000008, kd=0.0025, deadband=25):
        """PID controller — returns smoothed output given pixel error.
        deadband: errors smaller than this many pixels are treated as zero.
        Derivative is EMA-filtered (alpha=0.25) to reduce noise.
        Output EMA alpha=0.30.  Rate-limited to ±0.010 m/s per loop.
        """
        if abs(error) < deadband:
            error = 0  # ignore tiny wobble
        now = time.time()
        dt = (now - self._pid_last_time) if self._pid_last_time else 0.05
        self._pid_last_time = now
        self._pid_integral += error * dt
        self._pid_integral = max(-100, min(100, self._pid_integral))   # anti-windup
        raw_deriv = (error - self._pid_last_error) / max(dt, 1e-4)
        self._pid_last_error = error
        # Derivative EMA filter (alpha=0.25) — suppresses high-frequency noise
        self._deriv_smooth = 0.25 * raw_deriv + 0.75 * self._deriv_smooth
        raw = kp * error + ki * self._pid_integral + kd * self._deriv_smooth
        # Output EMA alpha=0.30
        self._vy_smooth = 0.30 * raw + 0.70 * self._vy_smooth
        # Rate limiter: max change of ±0.010 m/s per loop
        delta = self._vy_smooth - self._prev_vy
        delta = max(-0.010, min(0.010, delta))
        self._prev_vy = self._prev_vy + delta
        return self._prev_vy

    def line_follow(self, duration=30, forward_speed=0.2, land_on_tag=True, tag_ignore_secs=0):
        """
        PID line follower using the yellow line detected by the downward camera.
        Uses the bottom 40% of the frame as region-of-interest.
        tag_ignore_secs: suppress AprilTag detection for this many seconds at start.
        Returns the detected tag ID (int) if tag triggered exit, or None if timed out.
        """
        print(f"Line following for {duration}s  (forward={forward_speed} m/s, tag_ignore={tag_ignore_secs}s)")
        fw = 640  # frame width — will update from first frame
        tag_ignore_until = time.time() + tag_ignore_secs

        deadline = time.time() + duration
        while time.time() < deadline:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.02)
                continue

            fh, fw = frame.shape[:2]
            roi = frame[int(fh * 0.6):, :]          # bottom 40 % of frame

            # Detect yellow line in ROI using brightness threshold
            # (grayscale stream: yellow strip ≈ 170-226, dark floor ≈ 50-100)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))

            M = cv2.moments(mask)
            roi_y0 = int(fh * 0.6)
            # Use color-annotated frame as base so object overlays are visible
            disp = self._latest_display.copy() if self._latest_display is not None else frame.copy()

            # Draw ROI boundary
            cv2.rectangle(disp, (0, roi_y0), (fw - 1, fh - 1), (80, 80, 80), 1)
            cv2.putText(disp, "ROI", (4, roi_y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            if M['m00'] > 500:                      # line found
                cx = int(M['m10'] / M['m00'])       # centroid x in ROI
                error = cx - fw // 2
                vy = self._pid(error)
                vy = max(-0.15, min(0.15, vy))

                # Tint ROI with green mask overlay
                roi_tint = np.zeros_like(disp)
                roi_tint[roi_y0:, :] = cv2.merge([
                    np.zeros_like(mask), mask, np.zeros_like(mask)])  # green channel
                cv2.addWeighted(roi_tint, 0.25, disp, 1.0, 0, disp)

                # Vertical line at line centroid (full ROI height)
                cv2.line(disp, (cx, roi_y0), (cx, fh), (0, 255, 255), 2)

                # Frame centre vertical guide
                cv2.line(disp, (fw//2, roi_y0), (fw//2, fh), (100, 100, 100), 1)

                # Error bar — horizontal arrow from centre to line
                bar_y = roi_y0 + (fh - roi_y0) // 2
                err_color = (0, 200, 255) if abs(error) < 20 else \
                            (0, 140, 255) if abs(error) < 40 else (0, 60, 255)
                cv2.arrowedLine(disp, (fw//2, bar_y), (cx, bar_y), err_color, 2, tipLength=0.15)
                cv2.circle(disp, (cx, bar_y), 5, err_color, -1)

                # HUD panel — line info
                hud = [(f"Line  x={cx}", (0, 255, 255)),
                       (f"Err  {error:+d} px",  err_color),
                       (f"Vy   {vy:+.3f} m/s", (200, 255, 200))]
                for i, (txt, col) in enumerate(hud):
                    yy = 48 + i * 22
                    cv2.rectangle(disp, (0, yy - 16), (210, yy + 4), (0, 0, 0), -1)
                    cv2.putText(disp, txt, (6, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
            else:                                   # line lost
                vy = 0.0
                cv2.rectangle(disp, (0, 40), (160, 68), (0, 0, 180), -1)
                cv2.putText(disp, "LINE LOST", (6, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)

            # ── AprilTag detection ──────────────────────────────────────────
            if time.time() >= tag_ignore_until:
                tag = self._detect_apriltag(frame)
            else:
                tag = None

            if tag is not None:
                tid, tcx, tcy, tarea, tcorners = tag
                print(f"[TAG] AprilTag ID={tid}  area={tarea:.0f}  center=({tcx},{tcy})")

                # Draw tag outline + filled corners
                cv2.aruco.drawDetectedMarkers(disp, [tcorners], np.array([[tid]]))
                for pt in tcorners[0].astype(int):
                    cv2.circle(disp, tuple(pt), 5, (0, 255, 0), -1)

                # Centre rings
                cv2.circle(disp, (tcx, tcy), 8,  (0, 255, 0), 2)
                cv2.circle(disp, (tcx, tcy), 16, (0, 255, 0), 1)
                cv2.drawMarker(disp, (tcx, tcy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 1)

                # Tag label box
                tag_label = f" TAG {tid}  {tarea:.0f}px "
                (tlw, tlh), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(disp, (tcx - tlw//2 - 2, tcy - 36),
                                    (tcx + tlw//2 + 2, tcy - 36 + tlh + 8), (0, 80, 0), -1)
                cv2.putText(disp, tag_label, (tcx - tlw//2, tcy - 36 + tlh),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if tarea >= self._tag_land_area:
                    # Flash red border
                    cv2.rectangle(disp, (2, 2), (fw-2, fh-2), (0, 0, 255), 4)
                    cv2.rectangle(disp, (0, fh-32), (fw, fh), (0, 0, 180), -1)
                    cv2.putText(disp, "LANDING ON TAG", (fw//2 - 100, fh - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    self._push_display(disp)
                    print(f"[TAG] Tag large enough (area={tarea:.0f}) — stopping.")
                    self.control.set_velocity(0, 0, 0)
                    time.sleep(0.3)
                    if land_on_tag:
                        print("[TAG] Preparing to land on box...")
                        self._center_on_box()
                    return tid

            self._push_display(disp)
            self.control.set_velocity(vx=forward_speed, vy=vy, vz=0)

        # Stop movement when done
        self.control.set_velocity(0, 0, 0)
        print("Line following complete.")
        return None  # timed out, no tag triggered

    def _detect_box_center(self, frame):
        """
        Detect the landing pad by its bright WHITE edge.
        Finds the largest 4-corner quadrilateral from the white contour,
        computes geometric center as average of the 4 corners.
        Returns (cx, cy, pts_4x2, area) or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # High threshold — only grab the bright white box edge
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # Sort all contours largest-first
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                break   # already sorted, nothing bigger coming
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                sides = [np.linalg.norm(pts[i] - pts[(i+1) % 4])
                         for i in range(4)]
                if max(sides) == 0:
                    continue
                if min(sides) / max(sides) > 0.45:   # roughly square
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    return (cx, cy, pts, area)
        return None

    def _center_on_box(self, timeout=60.0):
        """
        Lock on the AprilTag / white-edge box center and land immediately.

        Logic:
          - Nudge slightly forward (NUDGE_FWD) every frame to keep the pad
            from drifting behind the drone.
          - Correct left/right with proportional control (no backward allowed).
          - After 3 consecutive frames within LOCK_THRESH px → locked, descend.
          - During descent keep correcting laterally all the way down.
        """
        self._box_landed = False
        LOCK_THRESH  = 25    # px — acceptable centering error
        STABLE_NEED  = 3     # consecutive good frames to call it locked
        NUDGE_FWD    = 0.03  # m/s constant tiny forward push while centering

        # Short stop — just enough to kill momentum
        self.control.set_velocity(0, 0, 0)
        time.sleep(0.5)

        def _get_error(frame):
            fh, fw = frame.shape[:2]
            tag = self._detect_apriltag(frame)
            if tag is not None:
                tid, tcx, tcy, tarea, tcorners = tag
                return tcx - fw // 2, tcy - fh // 2, 'tag', (tcx, tcy, tcorners, tid)
            result = self._detect_box_center(frame)
            if result is not None:
                bcx, bcy, pts, _ = result
                return bcx - fw // 2, bcy - fh // 2, 'box', (bcx, bcy, pts)
            return None

        # ── Phase 1: center & lock ────────────────────────────────────────
        print("[LAND] Locking onto pad...")
        stable = 0
        deadline = time.time() + 15.0   # fast timeout — don't hang forever

        while time.time() < deadline:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.03)
                continue

            fh, fw = frame.shape[:2]
            disp   = frame.copy()
            det    = _get_error(frame)

            if det is not None:
                ex, ey, source, info = det

                # Draw target
                if source == 'tag':
                    tcx, tcy, tcorners, tid = info
                    cv2.aruco.drawDetectedMarkers(disp, [tcorners], np.array([[tid]]))
                    cv2.circle(disp, (tcx, tcy), 10, (0, 255, 0), 2)
                    cv2.drawMarker(disp, (tcx, tcy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                else:
                    bcx, bcy, pts = info
                    for i in range(4):
                        cv2.line(disp, tuple(pts[i]), tuple(pts[(i+1) % 4]), (255, 0, 255), 2)
                    cv2.drawMarker(disp, (bcx, bcy), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

                cv2.drawMarker(disp, (fw // 2, fh // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
                col = (0, 255, 0) if abs(ex) < LOCK_THRESH and abs(ey) < LOCK_THRESH else (0, 180, 255)
                cv2.rectangle(disp, (0, 0), (fw, 32), (0, 0, 0), -1)
                cv2.putText(disp, f"[{source}] err=({ex:+d},{ey:+d})  lock={stable}/{STABLE_NEED}",
                            (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

                if abs(ex) < LOCK_THRESH and abs(ey) < LOCK_THRESH:
                    stable += 1
                    self.control.set_velocity(NUDGE_FWD, 0, 0)   # tiny fwd creep while locked
                    if stable >= STABLE_NEED:
                        print(f"[LAND] LOCKED [{source}] err=({ex:+d},{ey:+d}) — descending")
                        self._push_display(disp)
                        break
                else:
                    stable = 0
                    # Lateral correction + constant tiny forward nudge
                    vy_c = max(-0.10, min(0.10, ex * 0.0030))
                    vx_c = max(NUDGE_FWD, min(0.12, NUDGE_FWD + ey * 0.0030))  # fwd-only
                    self.control.set_velocity(vx=vx_c, vy=vy_c, vz=0)
            else:
                stable = 0
                cv2.rectangle(disp, (0, 0), (fw, 32), (0, 0, 0), -1)
                cv2.putText(disp, "LAND: searching...", (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1)
                # Creep forward so pad comes into view
                self.control.set_velocity(NUDGE_FWD, 0, 0)

            self._push_display(disp)
            time.sleep(0.04)

        # Full stop — kill all momentum before descending
        self.control.set_velocity(0, 0, 0)
        time.sleep(0.3)

        # ── Phase 2: straight vertical descent — no lateral movement ──────
        print("[LAND] Descending straight down (locked, no corrections)...")
        t_end  = time.time() + timeout
        last_alt = 3.0

        while time.time() < t_end:
            alt_msg = self.control.master.recv_match(
                type='GLOBAL_POSITION_INT', blocking=False)
            if alt_msg:
                last_alt = alt_msg.relative_alt / 1000.0

            if last_alt <= 0.15:
                print(f"[LAND] {last_alt:.2f} m — LAND command.")
                self.control.set_velocity(0, 0, 0)
                self.control.land()
                self._box_landed = True
                return

            vz = 0.35 if last_alt > 1.50 else \
                 0.20 if last_alt > 0.80 else \
                 0.10 if last_alt > 0.40 else 0.05

            # Pure vertical — no vx, no vy
            self.control.set_velocity(vx=0, vy=0, vz=vz)

            # Display only
            frame = self._latest_frame
            if frame is not None:
                disp = frame.copy()
                fh, fw = disp.shape[:2]
                cv2.drawMarker(disp, (fw // 2, fh // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
                cv2.rectangle(disp, (0, 0), (280, 28), (0, 0, 0), -1)
                cv2.putText(disp, f"ALT {last_alt:.2f}m  vz={vz:.2f}  LOCKED",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
                self._push_display(disp)

            time.sleep(0.04)

        print("[LAND] Timeout — LAND command.")
        self.control.set_velocity(0, 0, 0)
        self.control.land()
        self._box_landed = True

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

    def _reset_pid(self):
        """Reset PID state (call before a new line-follow phase)."""
        self._pid_integral   = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time  = None
        self._vy_smooth      = 0.0
        self._deriv_smooth   = 0.0
        self._prev_vy        = 0.0

    # ── main sequence ────────────────────────────────────────────────────────
    def start(self):
        """Take off, confirm camera feed, hold for 10 seconds, then land."""
        print("MAVLink connected. Starting simplified flight sequence...")

        # 1. Set GUIDED mode
        self.control.set_mode('GUIDED')

        # 2. Force arm
        self.control.force_arm()

        # 3. Takeoff
        target_alt_m = 2.2
        self.control.takeoff(target_alt_m)

        # 4. Start camera and wait for first frame
        cv2.namedWindow('Drone Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Camera', 640, 480)
        self.camera.start_thread(self.process_frame)
        print("Camera started. Waiting for first frame...")
        while self._latest_frame is None:
            time.sleep(0.05)

        print("Camera feed received. Landing in 10 seconds...")

        # 5. Keep preview active for 10 seconds before landing
        land_at = time.time() + 10.0
        while time.time() < land_at:
            frame = self._latest_display if self._latest_display is not None else self._latest_frame
            if frame is not None:
                self._push_display(frame.copy())
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