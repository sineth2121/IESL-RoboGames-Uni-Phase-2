'''
Minimal flight flow:
1) take off
2) wait for camera feed
3) detect square
4) center square
5) check yellow junction lines
6) line follow until black square appears
7) center black square
8) land
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
import math

# Percentage of strip resolution kept for sensor processing (1-100).
STRIP_DOWNSAMPLE_PERCENT = 20
# Check more frequently so fast motion does not skip the landing target.
FULL_FRAME_BLACK_SCAN_INTERVAL = 5
TARGET_LOST_ASCEND_INTERVAL_SECONDS = 0.8
TARGET_LOST_ASCEND_SPEED_MPS = -0.08
TARGET_LOST_ASCEND_DURATION_SECONDS = 0.25
APRILTAG_SIZE_METERS = 0.10
APRILTAG_YAW_SMOOTH_ALPHA = 0.85
APRILTAG_CENTER_SPEED_SCALE = 1.56
APRILTAG_ALIGN_TOLERANCE_DEG = 6.0
APRILTAG_ALIGN_HOLD_SECONDS = 0.8
APRILTAG_YAW_CMD_INTERVAL_SECONDS = 0.20
APRILTAG_YAW_MAX_STEP_DEG = 8.0
APRILTAG_YAW_MIN_STEP_DEG = 1.0
APRILTAG_IMAGE_ANGLE_SMOOTH_ALPHA = 0.80
# +1 means use raw error sign, -1 means inverted sign.
APRILTAG_ALIGN_INITIAL_YAW_SIGN = 1.0
TAG_REACQUIRE_BLOCK_SECONDS = 2.0

# Line-follow tuning globals.
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
        self._last_target_lost_ascend_time = 0.0
        self._tag_yaw_filtered_deg = None
        self._warned_uncalibrated_intrinsics = False
        self._latest_tag_info = None
        self._last_tag_yaw_cmd_time = 0.0
        self._tag_image_angle_filtered_deg = None
        self._align_yaw_sign = APRILTAG_ALIGN_INITIAL_YAW_SIGN
        self._last_align_error_abs = None

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

    def detect_black_square_and_print_each_frame(self, frame):
        """Detect landing box with AprilTag-like pattern on top using shape + darkness."""
        annotated = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        self._latest_tag_info = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Prefer explicit AprilTag-36h11 detection when available.
        tag_markers = self._detect_apriltag_36h11_markers(gray)
        if tag_markers:
            tag = min(
                tag_markers,
                key=lambda m: abs(m["center"][0] - frame_w / 2.0) + abs(m["center"][1] - frame_h / 2.0),
            )
            tag_box = tag["box"]
            tag_id = tag["id"]
            tag_corners = tag["corners"]

            camera_matrix, dist_coeffs = self._get_camera_intrinsics(frame.shape)
            yaw_deg = None
            if camera_matrix is not None and dist_coeffs is not None:
                # Object point order must match image corner order: TL, TR, BR, BL.
                s = APRILTAG_SIZE_METERS
                object_points = np.array([
                    [-s / 2.0, -s / 2.0, 0.0],
                    [ s / 2.0, -s / 2.0, 0.0],
                    [ s / 2.0,  s / 2.0, 0.0],
                    [-s / 2.0,  s / 2.0, 0.0],
                ], dtype=np.float32)

                image_points = tag_corners.astype(np.float32)
                solved, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if solved:
                    R, _ = cv2.Rodrigues(rvec)
                    yaw_rad = math.atan2(R[1, 0], R[0, 0])
                    if math.isfinite(yaw_rad):
                        yaw_deg_raw = self._wrap_angle_deg(math.degrees(yaw_rad))
                        yaw_deg = self._smooth_yaw_deg(yaw_deg_raw)
                    else:
                        yaw_deg = None

                    # Optional pose visualization.
                    if np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
                        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, APRILTAG_SIZE_METERS * 0.6)

            # Yellow path side detection relative to tag bounding box.
            connection_side, yellow_contours = self._detect_yellow_connection_side(frame, tag_box)
            if yellow_contours:
                cv2.drawContours(annotated, yellow_contours, -1, (0, 255, 255), 2)

            x, y, w, h = tag_box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.circle(annotated, tag["center"], 4, (255, 0, 255), -1)
            if yaw_deg is not None:
                self._draw_yaw_arrow(annotated, tag["center"], yaw_deg, length=36, color=(0, 200, 255))

            # Orientation from both parallel edges, smoothed in modulo-180 space.
            image_angle_deg = self._compute_tag_image_angle_deg(tag_corners)
            image_angle_filt_deg = self._smooth_image_angle_deg(image_angle_deg)
            align_error_deg = self._horizontal_alignment_error_deg(image_angle_filt_deg)
            self._latest_tag_info = {
                "id": tag_id,
                "box": tag_box,
                "center": tag["center"],
                "yaw_deg": yaw_deg,
                "image_angle_deg": image_angle_deg,
                "image_angle_filt_deg": image_angle_filt_deg,
                "align_error_deg": align_error_deg,
            }

            side_text = connection_side if connection_side is not None else "NONE"
            yaw_text = f"{yaw_deg:+.1f}" if yaw_deg is not None else "N/A"
            cv2.putText(
                annotated,
                f"Tag {tag_id} yaw={yaw_text}deg side={side_text} imgAng={image_angle_filt_deg:+.1f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
            )
            print(f"[APRILTAG] id={tag_id} yaw_deg={yaw_text} side={side_text} img_angle_raw={image_angle_deg:+.1f} img_angle_filt={image_angle_filt_deg:+.1f} align_err={align_error_deg:+.1f}")
            return annotated, tag_box

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Candidate mask allows dark and mixed dark/white (AprilTag-like) content.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (179, 170, 120))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        # Edge-based contour extraction is more robust when top face is not pure black.
        edges = cv2.Canny(blur, 40, 130)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_square = None
        best_score = float('-inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200:
                continue
            if area > frame_area * 0.45:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h if h else 1)
            # Relax aspect ratio: perspective can make the top face look rectangular.
            if not (0.55 <= aspect <= 1.80):
                continue

            roi_dark = dark_mask[y:y + h, x:x + w]
            if roi_dark.size == 0:
                continue

            # Require some dark support, but not fully dark, to allow AprilTag pattern.
            dark_ratio = float(np.count_nonzero(roi_dark)) / float(roi_dark.size)
            if dark_ratio < 0.15:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            dist_to_center = abs(cx - frame_w / 2.0) + abs(cy - frame_h / 2.0)

            # Favor larger centered squares with a moderate amount of dark area.
            darkness_bonus = 2200.0 * min(1.0, dark_ratio)
            score = area - 6.0 * dist_to_center + darkness_bonus
            if score > best_score:
                best_score = score
                best_square = (x, y, w, h)

        if best_square is not None:
            x, y, w, h = best_square
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(annotated, "Landing box detected", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            print("[VISION] Landing box detected")
        else:
            cv2.putText(annotated, "Landing box not detected", (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print("[VISION] Landing box not detected")

        return annotated, best_square

    def _detect_apriltag_36h11_markers(self, gray_frame):
        """Return detected 36h11 markers with id/corners/box/center."""
        if gray_frame is None or not hasattr(cv2, "aruco"):
            return []

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray_frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=params)

        if ids is None or len(ids) == 0:
            return []

        markers = []
        for marker_idx, marker_id in enumerate(ids.reshape(-1)):
            pts = corners[marker_idx].reshape(-1, 2).astype(np.float32)
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            markers.append({
                "id": int(marker_id),
                "corners": pts,
                "box": (x, y, w, h),
                "center": (cx, cy),
            })
        return markers

    def _get_camera_intrinsics(self, frame_shape):
        """Return camera intrinsics; prefers calibrated values if provided on this object."""
        if hasattr(self, "camera_matrix") and hasattr(self, "dist_coeffs"):
            return self.camera_matrix, self.dist_coeffs

        h, w = frame_shape[:2]
        fx = 0.9 * w
        fy = 0.9 * w
        cx = w / 2.0
        cy = h / 2.0
        camera_matrix = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        if not self._warned_uncalibrated_intrinsics:
            print("[APRILTAG] Using approximate intrinsics fallback. Set calibrated camera_matrix/dist_coeffs for best yaw accuracy.")
            self._warned_uncalibrated_intrinsics = True
        return camera_matrix, dist_coeffs

    def _wrap_angle_deg(self, angle_deg):
        """Normalize angle to [-180, 180]."""
        return ((angle_deg + 180.0) % 360.0) - 180.0

    def _smooth_yaw_deg(self, current_yaw_deg):
        """Low-pass yaw smoothing in angular space."""
        if not math.isfinite(current_yaw_deg):
            return None

        if self._tag_yaw_filtered_deg is None:
            self._tag_yaw_filtered_deg = self._wrap_angle_deg(current_yaw_deg)
            return self._tag_yaw_filtered_deg

        if not math.isfinite(self._tag_yaw_filtered_deg):
            self._tag_yaw_filtered_deg = self._wrap_angle_deg(current_yaw_deg)
            return self._tag_yaw_filtered_deg

        delta = self._wrap_angle_deg(current_yaw_deg - self._tag_yaw_filtered_deg)
        self._tag_yaw_filtered_deg = self._wrap_angle_deg(
            self._tag_yaw_filtered_deg + (1.0 - APRILTAG_YAW_SMOOTH_ALPHA) * delta
        )
        return self._tag_yaw_filtered_deg

    def _draw_yaw_arrow(self, frame, center, yaw_deg, length=32, color=(0, 200, 255)):
        """Draw heading arrow from tag center using filtered yaw."""
        if yaw_deg is None or not math.isfinite(yaw_deg):
            return

        cx, cy = center
        ang = math.radians(yaw_deg)
        if not math.isfinite(ang):
            return

        end_x = int(cx + length * math.cos(ang))
        end_y = int(cy + length * math.sin(ang))
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.25)

    def _compute_tag_image_angle_deg(self, corners):
        """Robust image angle from top+bottom tag edges in modulo-180 degrees."""
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        top_deg = math.degrees(math.atan2(float(top_right[1] - top_left[1]), float(top_right[0] - top_left[0])))
        bottom_deg = math.degrees(math.atan2(float(bottom_right[1] - bottom_left[1]), float(bottom_right[0] - bottom_left[0])))

        # Average parallel-edge orientations using double-angle trick (handles 180deg symmetry).
        t2 = math.radians(2.0 * top_deg)
        b2 = math.radians(2.0 * bottom_deg)
        avg2 = math.atan2(math.sin(t2) + math.sin(b2), math.cos(t2) + math.cos(b2))
        avg_deg = math.degrees(avg2) / 2.0
        return ((avg_deg + 90.0) % 180.0) - 90.0

    def _smooth_image_angle_deg(self, current_angle_deg):
        """Low-pass smoothing in modulo-180 angular space."""
        if self._tag_image_angle_filtered_deg is None:
            self._tag_image_angle_filtered_deg = ((current_angle_deg + 90.0) % 180.0) - 90.0
            return self._tag_image_angle_filtered_deg

        prev = self._tag_image_angle_filtered_deg
        delta = ((current_angle_deg - prev + 90.0) % 180.0) - 90.0
        prev = prev + (1.0 - APRILTAG_IMAGE_ANGLE_SMOOTH_ALPHA) * delta
        self._tag_image_angle_filtered_deg = ((prev + 90.0) % 180.0) - 90.0
        return self._tag_image_angle_filtered_deg

    def _horizontal_alignment_error_deg(self, angle_deg):
        """Error to horizontal alignment with 180deg symmetry, range [-90, 90]."""
        return ((angle_deg + 90.0) % 180.0) - 90.0

    def _send_relative_yaw_command(self, yaw_delta_deg, yaw_rate_dps=35.0):
        """Send small relative yaw command; positive is clockwise."""
        if yaw_delta_deg == 0.0:
            return
        self.control.master.mav.command_long_send(
            self.control.master.target_system,
            self.control.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            abs(yaw_delta_deg),
            yaw_rate_dps,
            1 if yaw_delta_deg > 0 else -1,
            1,
            0, 0, 0,
        )

    def _box_center_error(self, box, frame_shape):
        """Return center error tuple (err_x, err_y) in pixels."""
        x, y, w, h = box
        frame_h, frame_w = frame_shape[:2]
        cx = x + w / 2.0
        cy = y + h / 2.0
        return cx - (frame_w / 2.0), cy - (frame_h / 2.0)

    def _detect_yellow_connection_side(self, frame, tag_box):
        """Find yellow path side relative to tag bbox; returns (side, contours)."""
        x, y, w, h = tag_box
        tag_cx = x + w / 2.0
        tag_cy = y + h / 2.0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array((20, 100, 100), dtype=np.uint8)
        upper_yellow = np.array((35, 255, 255), dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, []

        best_side = None
        best_dist = float("inf")
        valid_contours = []
        min_area = 120
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            side = None
            if cx < x:
                side = "LEFT"
            elif cx > x + w:
                side = "RIGHT"
            elif cy < y:
                side = "TOP"
            elif cy > y + h:
                side = "BOTTOM"

            if side is None:
                continue

            valid_contours.append(cnt)
            dist = math.hypot(cx - tag_cx, cy - tag_cy)
            if dist < best_dist:
                best_dist = dist
                best_side = side

        return best_side, valid_contours

    def _analyze_yellow_directions_around_tag(self, frame, tag_box):
        """Crop a square ROI around the tag and report yellow at top/bottom/left/right centers."""
        annotated = frame.copy()
        x, y, w, h = tag_box
        frame_h, frame_w = frame.shape[:2]

        cx = x + w // 2
        cy = y + h // 2
        half = int(max(w, h) * 1.25)
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(frame_w, cx + half)
        y1 = min(frame_h, cy + half)

        if x1 <= x0 or y1 <= y0:
            print("[YELLOW_DIR] Invalid ROI around tag")
            return annotated, []

        roi = frame[y0:y1, x0:x1]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array((20, 100, 100), dtype=np.uint8)
        upper_yellow = np.array((35, 255, 255), dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        roi_h, roi_w = yellow_mask.shape[:2]
        sample_r = max(3, int(min(roi_h, roi_w) * 0.04))
        samples = {
            "TOP": (roi_w // 2, max(sample_r + 1, int(roi_h * 0.15))),
            "BOTTOM": (roi_w // 2, min(roi_h - sample_r - 1, int(roi_h * 0.85))),
            "LEFT": (max(sample_r + 1, int(roi_w * 0.15)), roi_h // 2),
            "RIGHT": (min(roi_w - sample_r - 1, int(roi_w * 0.85)), roi_h // 2),
        }

        active_dirs = []
        for name, (sx, sy) in samples.items():
            px0 = max(0, sx - sample_r)
            py0 = max(0, sy - sample_r)
            px1 = min(roi_w, sx + sample_r + 1)
            py1 = min(roi_h, sy + sample_r + 1)
            patch = yellow_mask[py0:py1, px0:px1]
            yellow_ratio = float(np.count_nonzero(patch)) / float(max(1, patch.size))
            is_yellow = yellow_ratio >= 0.22
            if is_yellow:
                active_dirs.append(name)

            fx = x0 + sx
            fy = y0 + sy
            color = (0, 255, 255) if is_yellow else (0, 0, 255)
            cv2.circle(annotated, (fx, fy), sample_r + 2, color, 2)
            cv2.putText(
                annotated,
                name,
                (fx + 6, fy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 1)
        dirs_text = ", ".join(active_dirs) if active_dirs else "NONE"
        print(f"[YELLOW_DIR] Around tag: {dirs_text}")
        cv2.putText(
            annotated,
            f"Yellow dirs: {dirs_text}",
            (8, 182),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        return annotated, active_dirs

    def _choose_next_direction(self, yellow_dirs):
        """Priority rule: RIGHT, then FRONT(TOP), then LEFT; fallback to FRONT."""
        dirs = set(yellow_dirs)
        if "RIGHT" in dirs:
            return "RIGHT"
        if "TOP" in dirs:
            return "TOP"
        if "LEFT" in dirs:
            return "LEFT"
        return "TOP"

    def _apply_direction_turn(self, direction):
        """Apply turn command for selected direction relative to current orientation."""
        if direction == "RIGHT":
            self.control.turn_yaw(90)
            print("[NAV] Turned RIGHT (+90 deg)")
        elif direction == "LEFT":
            self.control.turn_yaw(-90)
            print("[NAV] Turned LEFT (-90 deg)")
        else:
            print("[NAV] Going FRONT (no yaw turn)")

    def decode_apriltag_and_print(self, frame):
        """Decode AprilTag IDs from frame and print them. Returns list[int]."""
        if frame is None:
            print("[APRILTAG] No frame available for decoding")
            return []

        if not hasattr(cv2, "aruco"):
            print("[APRILTAG] OpenCV aruco module not available")
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers = self._detect_apriltag_36h11_markers(gray)
        detected_ids = [m["id"] for m in markers]
        if detected_ids:
            unique_ids = sorted(set(detected_ids))
            print(f"[APRILTAG] 36h11 IDs: {unique_ids}")
            return unique_ids

        print("[APRILTAG] No AprilTag 36h11 detected")
        return []

    def center_to_box(self, frame, box, speed_scale=1.0):
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
            vy = sign_x * (0.0155792 * gain_x * speed_scale)
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
            base_vx = sign_y * (0.0155792 * gain_y * speed_scale)
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
        forward_speed_nominal = 0.18
        forward_speed_min = 0.06
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
        error_mag = abs(error)
        if error_mag <= 0.12:
            vx = forward_speed_nominal
        else:
            # Larger line offset -> slow down forward motion for safer correction.
            slowdown = 1.0 - min(1.0, (error_mag - 0.12) / 0.88)
            vx = float(np.clip(
                forward_speed_nominal * (0.35 + 0.65 * slowdown),
                forward_speed_min,
                forward_speed_nominal,
            ))
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
            f"vx={vx:+.3f} vy={vy:+.3f} clipV={int(vy_clipped)} yawRaw={yaw_unclipped:+.2f} yaw={yaw_cmd_deg:+.2f} clipY={int(yaw_clipped)}"
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
        """Run mission phases and land after centering on the detected black square."""
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

        print("Color camera feed received. Starting sequence: line-follow -> landing-target center -> decode tag -> land")

        # 5. Run sequence phases
        # Temporary behavior: skip initial gray-box centering and begin line-follow immediately.
        phase = "line_follow"
        centered_since = None
        yellow_started_at = None
        black_centered_since = None
        align_centered_since = None
        line_follow_iter_count = 0
        center_hold_seconds = 1.2
        yellow_check_seconds = 2.0
        first_tag_id = None
        departed_tag_id = None
        departed_at = 0.0
        visited_non_first_tag = False
        last_processed_tag_id = None

        self._line_pid_integral = 0.0
        self._line_pid_prev_error = 0.0
        self._line_pid_prev_time = None

        while True:
            frame = self._latest_frame
            if frame is not None:
                analysis = frame.copy()
                detected_box = None

                if phase == "wait_square":
                    analysis, detected_box = self.detect_square_and_print_each_frame(frame.copy())
                    if detected_box is not None:
                        print("[PHASE] Square found -> centering phase")
                        phase = "center"
                    else:
                        print("[PHASE] Waiting for square...")

                elif phase == "center":
                    analysis, detected_box = self.detect_square_and_print_each_frame(frame.copy())
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
                    analysis, detected_box = self.detect_square_and_print_each_frame(frame.copy())
                    analysis, yellow_location = self.junction_logic(analysis, detected_box)

                    if yellow_location == "top-center":
                        phase = "line_follow"
                        line_follow_iter_count = 0
                        self._line_pid_integral = 0.0
                        self._line_pid_prev_error = 0.0
                        self._line_pid_prev_time = None
                        print("[PHASE] top-center detected -> line follow until landing box appears")

                    elapsed = time.time() - yellow_started_at if yellow_started_at else 0.0
                    print(f"[PHASE] Yellow check: {elapsed:.1f}/{yellow_check_seconds:.1f}s")
                    if elapsed >= yellow_check_seconds:
                        print("[PHASE] Yellow check complete -> landing")
                        break

                elif phase == "line_follow":
                    src = self._latest_frame if self._latest_frame is not None else frame
                    strip = self._get_top_strip(src)
                    analysis = self._line_follow_from_strip(strip)

                    line_follow_iter_count += 1
                    if line_follow_iter_count % FULL_FRAME_BLACK_SCAN_INTERVAL == 0:
                        print("[PHASE] Line follow check: scanning full frame for landing box")
                        full_analysis, black_box = self.detect_black_square_and_print_each_frame(src.copy())
                        if black_box is not None:
                            detected_tag_id = self._latest_tag_info["id"] if self._latest_tag_info is not None else None
                            if (
                                departed_tag_id is not None
                                and detected_tag_id == departed_tag_id
                                and (time.time() - departed_at) < TAG_REACQUIRE_BLOCK_SECONDS
                            ):
                                print(f"[NAV] Ignoring immediate reacquire of tag {detected_tag_id}")
                                analysis = full_analysis
                                continue

                            phase = "center_black"
                            black_centered_since = None
                            # Reset centering memory so prior target history does not bias this lock.
                            self._last_cmd_axis = None
                            self._last_err_y_abs = None
                            self._err_x_filt = 0.0
                            self._err_y_filt = 0.0
                            self._last_cmd_sign_x = 0
                            self._last_cmd_sign_y = 0
                            self._settle_until = 0.0
                            self._last_target_lost_ascend_time = 0.0
                            analysis = full_analysis
                            print(f"[PHASE] Landing box found (tag={detected_tag_id}) -> stop line follow, centering")
                        else:
                            print(f"[PHASE] Line follow ongoing... iter={line_follow_iter_count}")

                elif phase == "center_black":
                    analysis, black_box = self.detect_black_square_and_print_each_frame(frame.copy())
                    analysis, centered = self.center_to_box(
                        analysis,
                        black_box,
                        speed_scale=APRILTAG_CENTER_SPEED_SCALE,
                    )

                    if black_box is None:
                        # Keep line-follow stopped; nudge up a little to recover line-of-sight.
                        now = time.time()
                        if now - self._last_target_lost_ascend_time >= TARGET_LOST_ASCEND_INTERVAL_SECONDS:
                            self.control.move_with_velocity(
                                vx=0.0,
                                vy=0.0,
                                vz=TARGET_LOST_ASCEND_SPEED_MPS,
                                duration=TARGET_LOST_ASCEND_DURATION_SECONDS,
                            )
                            self._last_target_lost_ascend_time = now
                            print("[PHASE] Landing box lost -> small ascend and continue centering search")
                        black_centered_since = None
                    elif centered:
                        if black_centered_since is None:
                            black_centered_since = time.time()
                        if time.time() - black_centered_since >= center_hold_seconds:
                            phase = "align_tag"
                            align_centered_since = None
                            self._last_tag_yaw_cmd_time = 0.0
                            self._align_yaw_sign = APRILTAG_ALIGN_INITIAL_YAW_SIGN
                            self._last_align_error_abs = None
                            print("[PHASE] Landing box centered -> align tag orientation")
                    else:
                        black_centered_since = None

                elif phase == "align_tag":
                    analysis, black_box = self.detect_black_square_and_print_each_frame(frame.copy())
                    if black_box is None:
                        phase = "center_black"
                        align_centered_since = None
                        print("[PHASE] Lost tag during alignment -> back to centering")
                    else:
                        err_x, err_y = self._box_center_error(black_box, frame.shape)
                        if abs(err_x) > 35 or abs(err_y) > 35:
                            phase = "center_black"
                            align_centered_since = None
                            print("[PHASE] Tag drifted from center -> recentering")
                        else:
                            # Keep translation still while rotating in place.
                            self.control.move_with_velocity(vx=0.0, vy=0.0, vz=0.0, duration=0.04)

                            align_error = None
                            if self._latest_tag_info is not None:
                                align_error = self._latest_tag_info.get("align_error_deg")

                            if align_error is None:
                                print("[PHASE] Aligning tag: waiting for orientation estimate")
                                align_centered_since = None
                            elif abs(align_error) <= APRILTAG_ALIGN_TOLERANCE_DEG:
                                if align_centered_since is None:
                                    align_centered_since = time.time()
                                if time.time() - align_centered_since >= APRILTAG_ALIGN_HOLD_SECONDS:
                                    analysis, yellow_dirs = self._analyze_yellow_directions_around_tag(frame.copy(), black_box)
                                    decode_src = self._latest_frame if self._latest_frame is not None else frame
                                    detected_ids = self.decode_apriltag_and_print(decode_src)

                                    current_tag_id = None
                                    if self._latest_tag_info is not None:
                                        current_tag_id = self._latest_tag_info.get("id")
                                    elif detected_ids:
                                        current_tag_id = detected_ids[0]

                                    if current_tag_id is not None and first_tag_id is None:
                                        first_tag_id = current_tag_id
                                        print(f"[NAV] Stored first tag id={first_tag_id}")

                                    if (
                                        current_tag_id is not None
                                        and last_processed_tag_id is not None
                                        and current_tag_id == last_processed_tag_id
                                    ):
                                        print(f"[NAV] Consecutive same tag id={current_tag_id} -> ignore and continue line-follow")
                                        departed_tag_id = current_tag_id
                                        departed_at = time.time()
                                        phase = "line_follow"
                                        line_follow_iter_count = 0
                                        self._line_pid_integral = 0.0
                                        self._line_pid_prev_error = 0.0
                                        self._line_pid_prev_time = None
                                        continue

                                    if current_tag_id is not None and first_tag_id is not None and current_tag_id != first_tag_id:
                                        visited_non_first_tag = True

                                    if (
                                        current_tag_id is not None
                                        and first_tag_id is not None
                                        and visited_non_first_tag
                                        and current_tag_id == first_tag_id
                                    ):
                                        print(f"[NAV] Returned to first tag id={first_tag_id} -> landing")
                                        break

                                    next_dir = self._choose_next_direction(yellow_dirs)
                                    print(f"[NAV] Tag id={current_tag_id} yellow_dirs={yellow_dirs} -> next={next_dir}")
                                    self._apply_direction_turn(next_dir)

                                    departed_tag_id = current_tag_id
                                    departed_at = time.time()
                                    last_processed_tag_id = current_tag_id
                                    phase = "line_follow"
                                    line_follow_iter_count = 0
                                    self._line_pid_integral = 0.0
                                    self._line_pid_prev_error = 0.0
                                    self._line_pid_prev_time = None
                                    print("[PHASE] Resume line-follow to next tag")
                            else:
                                align_centered_since = None
                                now = time.time()
                                if now - self._last_tag_yaw_cmd_time >= APRILTAG_YAW_CMD_INTERVAL_SECONDS:
                                    if self._last_align_error_abs is not None and abs(align_error) > (self._last_align_error_abs + 2.0):
                                        self._align_yaw_sign *= -1.0
                                        print(f"[PHASE] Align direction flipped. sign={self._align_yaw_sign:+.0f}")

                                    yaw_cmd = self._align_yaw_sign * (0.7 * align_error)
                                    yaw_cmd = float(np.clip(yaw_cmd, -APRILTAG_YAW_MAX_STEP_DEG, APRILTAG_YAW_MAX_STEP_DEG))
                                    if 0.0 < abs(yaw_cmd) < APRILTAG_YAW_MIN_STEP_DEG:
                                        yaw_cmd = APRILTAG_YAW_MIN_STEP_DEG if yaw_cmd > 0 else -APRILTAG_YAW_MIN_STEP_DEG
                                    self._send_relative_yaw_command(yaw_cmd)
                                    self._last_tag_yaw_cmd_time = now
                                    self._last_align_error_abs = abs(align_error)
                                    print(f"[PHASE] Aligning tag orientation: err={align_error:+.1f} yaw_cmd={yaw_cmd:+.1f}")

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