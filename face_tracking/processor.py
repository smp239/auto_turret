# processor.py
"""Camera → detector → tracker → turret glue with live-tuning.

Changes 2025-06-22
------------------
* Added live-tuning: edit ``runtime_params.json`` while the program is running
  to adjust tracker noise terms (`predict_lead_time_s`, `measurement_noise_std`,
  `process_noise_std`) **and** turret controller gains / limits
  (`kp_pan`, `kp_tilt`, `error_deadzone_px`, `min_angle_change_deg`,
  `max_cmd_rate_hz`).  Updates take effect on the next frame.
* Startup: if `home_on_startup` is set, the turret is homed **and then** moved
  to its resting position.
* No-target logic: before the first valid detection the turret stays at rest.
* Rest throttle: prevents spamming `rest()` every frame.
"""
from __future__ import annotations

import time
import traceback
from typing import Optional

import cv2
import numpy as np
import serial

from face_tracking.camera import Camera
from face_tracking.common import TargetReport
from face_tracking.config import (
    CameraConfig,
    DetectorConfig,
    PanTiltConfig,
    SearchConfig,
    TrackerConfig,
)
from face_tracking.detector import MediaPipeFaceDetector
from face_tracking.helpers import DetectionSmoother
from face_tracking.live_tuning import RuntimeParamWatcher          # ← NEW
from face_tracking.pan_tilt import FirmwareError, PanTilt
from face_tracking.tracker import KalmanTracker


class TargetingProcessor:
    """The main high-level orchestrator."""
    _REST_THROTTLE_S = 5.0          # minimum time between successive rest()s

    # ------------------------------------------------------------------ #
    #   I N I T
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        camera_cfg: CameraConfig,
        detector_cfg: DetectorConfig,
        tracker_cfg: TrackerConfig,
        pantilt_cfg: PanTiltConfig,
        search_cfg: SearchConfig,
    ):
        # Config blobs --------------------------------------------------
        self.camera_cfg = camera_cfg
        self.detector_cfg = detector_cfg
        self.tracker_cfg = tracker_cfg
        self.pantilt_cfg = pantilt_cfg
        self.search_cfg = search_cfg

        # Core pipeline objects ----------------------------------------
        self.camera = Camera(camera_cfg)
        init_dt = (
            1.0 / camera_cfg.fps_request
            if camera_cfg.fps_request > 0 else
            1.0 / 30.0
        )
        self.tracker = KalmanTracker(tracker_cfg, init_dt)
        self.detector = MediaPipeFaceDetector(detector_cfg)
        self.smoother = DetectionSmoother(alpha=0.3)

        # Turret related -----------------------------------------------
        self.turret: Optional[PanTilt] = None
        self.deg_per_px_h = 0.0
        self.deg_per_px_v = 0.0

        self.last_turret_cmd = 0.0
        self.min_cmd_interval = (
            1.0 / pantilt_cfg.max_cmd_rate_hz
            if pantilt_cfg.max_cmd_rate_hz > 0 else
            0.0
        )

        # Stats & search ------------------------------------------------
        self.frame_count = 0
        self.proc_time_sum = 0.0
        self.proc_samples = 0
        self.fps_timer_start = time.time()
        self.disp_fps = 0.0
        self.disp_proc_ms_avg = 0.0

        self.search_active = False
        self.last_detection_time = time.time()
        self.search_idx = 0
        self.search_move_time = 0.0

        self.cam_reopens = 0
        self.max_cam_reopens = 5
        self.turret_ok = False
        self.last_turret_error = 0.0
        self.turret_cooldown = 15.0

        self.last_valid_frame: Optional[np.ndarray] = None
        self.total_frames = 0

        # State flags ---------------------------------------------------
        self.had_target = False
        self.last_rest_time = 0.0

        # Live-tuning ---------------------------------------------------
        self.param_watcher = RuntimeParamWatcher()        # ← NEW
        self._apply_runtime_params(initial=True)          # ← NEW

    # ------------------------------------------------------------------ #
    #   L I V E   T U N I N G
    # ------------------------------------------------------------------ #
    def _apply_runtime_params(self, *, initial: bool = False) -> None:
        """Push JSON parameters into tracker & pantilt config."""
        p = self.param_watcher.params

        # ---- tracker ----
        self.tracker.apply_tuning(
            predict_lead_time_s=p.get("predict_lead_time_s"),
            measurement_noise_std=p.get("measurement_noise_std"),
            process_noise_std=p.get("process_noise_std"),
        )

        # ---- pantilt cfg ----
        if "kp_pan" in p:
            self.pantilt_cfg.kp_pan = float(p["kp_pan"])
        if "kp_tilt" in p:
            self.pantilt_cfg.kp_tilt = float(p["kp_tilt"])
        if "error_deadzone_px" in p:
            self.pantilt_cfg.error_deadzone_px = int(p["error_deadzone_px"])
        if "min_angle_change_deg" in p:
            self.pantilt_cfg.min_angle_change_deg = float(
                p["min_angle_change_deg"]
            )
        if "max_cmd_rate_hz" in p and p["max_cmd_rate_hz"] > 0:
            self.pantilt_cfg.max_cmd_rate_hz = float(p["max_cmd_rate_hz"])
            self.min_cmd_interval = 1.0 / self.pantilt_cfg.max_cmd_rate_hz

        if not initial:
            print("[Runtime] Parameters updated.")

    # ------------------------------------------------------------------ #
    #   S E T U P / C L E A N U P
    # ------------------------------------------------------------------ #
    def setup(self) -> bool:
        if not self.camera.open():
            return False

        self.cam_reopens = 0
        w, h, fps, _, hfov, vfov = self.camera.get_properties()
        if fps > 0:
            self.tracker.update_nominal_dt(1.0 / fps)
        if hfov > 0 and w > 0:
            self.deg_per_px_h = hfov / w
        if vfov > 0 and h > 0:
            self.deg_per_px_v = vfov / h

        if self.pantilt_cfg.port:
            try:
                self.turret = PanTilt(
                    self.pantilt_cfg.port,
                    baudrate=self.pantilt_cfg.baudrate,
                )
                self.turret_ok = True
            except Exception as exc:                     # noqa: BLE001
                print(f"[Turret] Init error: {exc}")
                self.turret_ok = False

        cv2.namedWindow("Target Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "Target Tracking",
            self.camera_cfg.width // 2,
            self.camera_cfg.height // 2,
        )
        print("[Processor] Setup complete – press 'q' to quit.")
        return True

    def cleanup(self) -> None:
        print("[Processor] Cleaning up…")
        self.camera.release()
        self.detector.close()
        if self.turret and self.turret.is_open():
            try:
                self.turret.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print(f"[Processor] Exited. Total frames: {self.total_frames}")

    # ------------------------------------------------------------------ #
    #   D R A W I N G   U T I L S
    # ------------------------------------------------------------------ #
    def _draw_overlay(self, img: np.ndarray, rpt: Optional[TargetReport]) -> None:
        cv2.putText(
            img,
            f"FPS:{self.disp_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f"Proc:{self.disp_proc_ms_avg:.1f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if rpt and rpt.bbox_px:
            x, y, w, h = rpt.bbox_px
            x_i, y_i, w_i, h_i = map(int, (x, y, w, h))
            cv2.rectangle(
                img,
                (x_i, y_i),
                (x_i + w_i, y_i + h_i),
                (0, 255, 0),
                2,
            )
            label = (
                f"Cf:{rpt.confidence:.2f}"
                if rpt.confidence is not None
                else "Det"
            )
            cv2.putText(
                img,
                label,
                (x_i, y_i - 10 if y_i > 10 else y_i + h_i + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if rpt:
            cx_i, cy_i = map(int, rpt.kf_now_px)
            fx_i, fy_i = map(int, rpt.kf_future_px)
            cv2.circle(img, (cx_i, cy_i), 7, (0, 255, 255), -1)
            cv2.circle(img, (fx_i, fy_i), 9, (0, 0, 255), 2)
            cv2.putText(
                img,
                f"ID:{rpt.track_id} Age:{rpt.age_frames}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

        if self.search_active:
            cv2.putText(
                img,
                "SEARCHING",
                (img.shape[1] - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

    # ------------------------------------------------------------------ #
    #   T U R R E T   C O N T R O L
    # ------------------------------------------------------------------ #
    def _send_turret(self, pan: float, tilt: float) -> bool:
        if not (self.turret and self.turret.is_open() and self.turret_ok):
            return False
        now = time.time()
        if now - self.last_turret_cmd < self.min_cmd_interval:
            return False
        try:
            if self.pantilt_cfg.use_blocking_moves:
                self.turret.move_deg(pan, tilt)
            else:
                self.turret.async_move_deg(pan, tilt)
            self.last_turret_cmd = now
            return True
        except (FirmwareError, serial.SerialException, RuntimeError) as exc:
            print(f"[Turret] Cmd error: {exc}")
            self.turret_ok = False
            self.last_turret_error = now
            return False

    def _ensure_rest(self, now: float) -> None:
        """Send the rest command at most once every `_REST_THROTTLE_S`."""
        if (
            not self.turret_ok
            or not self.turret
            or not self.turret.is_open()
        ):
            return
        if now - self.last_rest_time < self._REST_THROTTLE_S:
            return
        try:
            self.turret.rest()
            self.last_rest_time = now
        except Exception as exc:
            print(f"[Turret] Rest failed: {exc}")

    def _update_turret(self, rpt: TargetReport) -> None:
        """Closed-loop tracking control."""
        if (
            not self.turret
            or not self.turret.is_open()
            or not self.turret_ok
            or self.deg_per_px_h == 0
            or self.deg_per_px_v == 0
        ):
            return

        img_w, img_h = rpt.img_size
        tx, ty = rpt.kf_future_px
        cx, cy = img_w / 2.0, img_h / 2.0
        ex, ey = tx - cx, ty - cy

        if abs(ex) < self.pantilt_cfg.error_deadzone_px:
            ex = 0.0
        if abs(ey) < self.pantilt_cfg.error_deadzone_px:
            ey = 0.0
        if ex == 0.0 and ey == 0.0:
            return

        pan_view = self.pantilt_cfg.kp_pan * (ex * self.deg_per_px_h)
        tilt_view = (
            self.pantilt_cfg.kp_tilt * (ey * self.deg_per_px_v) * -1.0
        )

        pan_cmd = pan_view * (
            -1.0 if self.pantilt_cfg.invert_pan_output else 1.0
        )
        tilt_cmd = tilt_view * (
            -1.0 if self.pantilt_cfg.invert_tilt_output else 1.0
        )

        if (
            abs(pan_cmd) < self.pantilt_cfg.min_angle_change_deg
            and abs(tilt_cmd) < self.pantilt_cfg.min_angle_change_deg
        ):
            return

        try:
            curr_pan, curr_tilt = self.turret.position_deg()
            self._send_turret(curr_pan + pan_cmd, curr_tilt + tilt_cmd)
        except Exception as exc:
            print(f"[Turret] Pos query error: {exc}")
            self.turret_ok = False
            self.last_turret_error = time.time()

    # ------------------------------------------------------------------ #
    #   S E A R C H / N O   D E T E C T I O N
    # ------------------------------------------------------------------ #
    def _handle_no_detection(self, now: float) -> None:
        """Rest or run search pattern depending on state."""
        if (
            not self.search_cfg.enabled
            or not (self.turret and self.turret.is_open() and self.turret_ok)
            or not self.search_cfg.pan_angles_deg
        ):
            self.search_active = False
            return

        # Haven’t seen any target yet → stay at rest
        if not self.had_target:
            self.search_active = False
            self._ensure_rest(now)
            return

        # Wait threshold before starting search
        if now - self.last_detection_time < self.search_cfg.no_detection_threshold_s:
            self.search_active = False
            return

        # Run search pattern
        if not self.search_active:
            self.search_active = True
            self.search_idx = 0
            self.search_move_time = now - self.search_cfg.dwell_time_s - 1.0

        if now - self.search_move_time >= self.search_cfg.dwell_time_s:
            angle = self.search_cfg.pan_angles_deg[self.search_idx]
            if self._send_turret(angle, self.search_cfg.tilt_deg):
                self.search_move_time = now
                self.search_idx = (self.search_idx + 1) % len(
                    self.search_cfg.pan_angles_deg
                )

    # ------------------------------------------------------------------ #
    #   F R A M E   P R O C E S S I N G
    # ------------------------------------------------------------------ #
    def _process_frame(self) -> bool:
        # -------- live-tuning hook ------------------------------------
        if self.param_watcher.maybe_reload():             # ← NEW
            self._apply_runtime_params()                  # ← NEW

        now = time.time()

        # Turret recovery after error
        if (
            self.turret
            and not self.turret_ok
            and now - self.last_turret_error > self.turret_cooldown
        ):
            try:
                if not self.turret.is_open():
                    self.turret.open()
                self.turret_ok = self.turret.is_open()
            except Exception:
                self.last_turret_error = now

        # Grab frame
        ts, frame = self.camera.read()
        if frame is None:
            # try to reopen camera a few times
            if (
                not self.camera.is_opened()
                and self.cam_reopens < self.max_cam_reopens
            ):
                if self.camera.open():
                    self.cam_reopens = 0
                    _, _, fps, _, _, _ = self.camera.get_properties()
                    if fps > 0:
                        self.tracker.update_nominal_dt(1.0 / fps)
                else:
                    self.cam_reopens += 1

            if self.last_valid_frame is not None:
                disp = self.last_valid_frame.copy()
                cv2.putText(
                    disp,
                    "Cam Err",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Target Tracking", disp)
            time.sleep(0.05)
            return True

        self.last_valid_frame = frame.copy()
        self.total_frames += 1

        # Ensure BGR
        if frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame

        # Detect & track
        tic = time.time()
        detections = self.detector.detect(bgr)

        meas = None
        bbox = None
        conf = None
        if detections:
            bbox, conf = detections[0]
            bbox = self.smoother.update(bbox)
            meas = (
                bbox[0] + bbox[2] / 2.0,
                bbox[1] + bbox[3] / 2.0,
            )

        self.tracker.predict_and_update(meas, ts, bbox, conf)
        rpt_src = self.tracker.get_report()
        rpt = None
        if rpt_src:
            rpt = TargetReport(
                t_capture=rpt_src.t_capture,
                kf_now_px=rpt_src.kf_now_px,
                kf_future_px=rpt_src.kf_future_px,
                velocity_px_s=rpt_src.velocity_px_s,
                bbox_px=rpt_src.bbox_px,
                img_size=(bgr.shape[1], bgr.shape[0]),
                confidence=rpt_src.confidence,
                age_frames=rpt_src.age_frames,
                track_id=rpt_src.track_id,
            )

        # Turret & search logic
        if rpt and rpt.age_frames > 2:
            self.had_target = True
            self.last_detection_time = now
            self.search_active = False
            self._update_turret(rpt)
        else:
            if self.turret_ok:
                self._handle_no_detection(now)
            else:
                self.search_active = False

        # Stats
        proc_ms = (time.time() - tic) * 1000.0
        self.proc_time_sum += proc_ms
        self.proc_samples += 1
        self.frame_count += 1
        if now - self.fps_timer_start >= 1.0:
            self.disp_fps = self.frame_count / (now - self.fps_timer_start)
            if self.proc_samples:
                self.disp_proc_ms_avg = self.proc_time_sum / self.proc_samples
            self.frame_count = 0
            self.proc_time_sum = 0.0
            self.proc_samples = 0
            self.fps_timer_start = now

        # Draw & display
        out = bgr.copy()
        self._draw_overlay(out, rpt)
        cv2.imshow("Target Tracking", out)
        return True

    # ------------------------------------------------------------------ #
    #   R U N   L O O P
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if not self.setup():
            self.cleanup()
            return

        turret_ctx = False
        try:
            if self.turret:
                with self.turret:
                    turret_ctx = True

                    # ---------- Startup homing + resting ----------------
                    if self.turret_ok and self.pantilt_cfg.home_on_startup:
                        try:
                            self.turret.home()
                        except Exception as exc:           # noqa: BLE001
                            print(f"[Turret] Homing failed: {exc}")
                            self.turret_ok = False
                            self.last_turret_error = time.time()

                    if self.turret_ok:
                        try:
                            self.turret.rest()
                            self.last_rest_time = time.time()
                        except Exception as exc:           # noqa: BLE001
                            print(f"[Turret] Rest failed: {exc}")

                    # ---------- Main loop --------------------------------
                    while True:
                        if not self._process_frame():
                            break
                        key = cv2.waitKey(15) & 0xFF
                        if key == ord("q"):
                            break
                        elif key == ord("h") and self.turret_ok:
                            try:
                                self.turret.home()
                                if self.turret_ok:
                                    self.turret.rest()
                                    self.last_rest_time = time.time()
                            except Exception as exc:
                                print(f"[Turret] Homing failed: {exc}")
                        elif key == ord("r") and self.turret_ok:
                            try:
                                self.turret.rest()
                                self.last_rest_time = time.time()
                            except Exception as exc:
                                print(f"[Turret] Rest failed: {exc}")
            else:
                # No turret – just show detections
                while True:
                    if not self._process_frame():
                        break
                    if (cv2.waitKey(15) & 0xFF) == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\n[Processor] Stopped by user.")
        except Exception as exc:                           # noqa: BLE001
            print(f"[Processor] Main loop error: {exc}")
            traceback.print_exc()
        finally:
            if turret_ctx and self.turret and self.turret.is_open():
                if self.pantilt_cfg.rest_on_shutdown and self.turret_ok:
                    try:
                        self.turret.rest()
                    except Exception:
                        pass
            self.cleanup()
