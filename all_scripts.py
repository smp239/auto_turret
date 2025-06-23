Thoroughly review and analyze the following codebase. Then determine if it would benefit from something like KCF or MOSSE? Are there better methods to lead a target than what is in the codebase? What further improvements can be made? Are the methods I'm using for moving the camera itself good or are there better alternatives? I plan to eventually use this for targeting an automatic paintball turret.

Note: All scripts but the Arduino Nano ESP32 script are being ran on Windows WSL Ubuntu.

main.py:
# main.py
"""
Entry-point for the face-tracking system.

Live-tuning
-----------
While the program is running you can edit ``runtime_params.json`` and the new
values (lead-time, noise terms, controller gains, etc.) will take effect on
the very next frame.  See ``face_tracking/live_tuning.py`` for details.

Extra (Windows / WSL2 only)
---------------------------
When executed inside WSL2 the script ensures that the required USB devices
(bus-ids listed in ``_REQUIRED_BUSIDS``) are attached via *usbipd* **before**
the vision stack starts.  Harmless on non-Windows hosts.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable, Set

from face_tracking.config import (
    CameraConfig,
    DetectorConfig,
    PanTiltConfig,
    SearchConfig,
    TrackerConfig,
)
from face_tracking.processor import TargetingProcessor

# ────────────────────────────────────────────────────────────────────────────
#   USBIPD  HELPER (only relevant on Windows/WSL2)
# ────────────────────────────────────────────────────────────────────────────
_REQUIRED_BUSIDS = ("6-1", "1-9")  # Update if you add/remove hardware


def _ensure_usb_devices(busids: Iterable[str]) -> None:
    """Detach + attach each bus-id so it shows up inside WSL."""
    usbipd_exe = shutil.which("usbipd.exe") or shutil.which("usbipd")
    if not usbipd_exe:  # not on Windows or usbipd not installed
        return

    try:
        cp = subprocess.run(
            [usbipd_exe, "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.SubprocessError as exc:
        print(f"[USBIPD] Could not run `usbipd list`: {exc}", file=sys.stderr)
        return

    shared: Set[str] = set()
    for line in cp.stdout.splitlines():
        m = re.match(r"^\s*(\d+-\d+)\s+.*\s+(Attached)\s*$", line)
        if m:
            shared.add(m.group(1))

    for bid in busids:
        if bid in shared:
            print(f"[USBIPD] Detaching bus {bid} from WSL …")
            try:
                subprocess.run([usbipd_exe, "detach", "-b", bid], check=True)
                time.sleep(1)
            except subprocess.SubprocessError as exc:
                print(f"[USBIPD]   ! failed to detach {bid}: {exc}", file=sys.stderr)

        print(f"[USBIPD] Attaching bus {bid} to WSL …")
        try:
            subprocess.run([usbipd_exe, "attach", "-b", bid, "--wsl"], check=True)
            time.sleep(2)
        except subprocess.SubprocessError as exc:
            print(f"[USBIPD]   ! failed to attach {bid}: {exc}", file=sys.stderr)


def detach_usb_devices(busids: Iterable[str]) -> None:
    """Detach the given USB bus-ids from WSL on shutdown (best-effort)."""
    usbipd_exe = shutil.which("usbipd.exe") or shutil.which("usbipd")
    if not usbipd_exe:
        return

    try:
        cp = subprocess.run(
            [usbipd_exe, "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.SubprocessError as exc:
        print(f"[USBIPD] Could not run `usbipd list`: {exc}", file=sys.stderr)
        return

    attached: Set[str] = set()
    for line in cp.stdout.splitlines():
        m = re.match(r"^\s*(\d+-\d+)\s+.*\s+(Attached)\s*$", line)
        if m:
            attached.add(m.group(1))

    for bid in busids:
        if bid not in attached:
            continue
        print(f"[USBIPD] Detaching bus {bid} from WSL …")
        try:
            subprocess.run([usbipd_exe, "detach", "-b", bid], check=True)
            time.sleep(1)
        except subprocess.SubprocessError as exc:
            print(f"[USBIPD]   ! failed to detach {bid}: {exc}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────
#   M A I N
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _ensure_usb_devices(_REQUIRED_BUSIDS)

    print("Initializing Face-Tracking System…")
    print("Hint: edit 'runtime_params.json' at any time to tweak parameters.\n")

    # -------------------- Config blobs --------------------
    cam_cfg  = CameraConfig()
    det_cfg  = DetectorConfig()
    trk_cfg  = TrackerConfig()
    pt_cfg   = PanTiltConfig(port="/dev/ttyACM0")  # set to None to disable turret
    srch_cfg = SearchConfig(
        enabled=True,
        pan_angles_deg=[45.0, 0.0, -40.0, 0.0],
        dwell_time_s=2.5,
        no_detection_threshold_s=3.5,
    )

    # ------------------------ Banner ----------------------
    print(
        f"Camera: idx={cam_cfg.device_index}, "
        f"{cam_cfg.width}x{cam_cfg.height}@{cam_cfg.fps_request} FPS, "
        f"HFOV={cam_cfg.horizontal_fov_deg}°"
    )
    print(
        f"Detector: conf={det_cfg.min_detection_confidence}, "
        f"min_bbox={det_cfg.min_bbox_size_for_tracking}px"
    )
    if pt_cfg.port:
        print(
            f"PanTilt: port={pt_cfg.port}, "
            f"Kp=({pt_cfg.kp_pan},{pt_cfg.kp_tilt}), "
            f"deadzone={pt_cfg.error_deadzone_px}px, "
            f"rate={pt_cfg.max_cmd_rate_hz} Hz"
        )
        print(
            f"Search: enabled={srch_cfg.enabled}, "
            f"angles={srch_cfg.pan_angles_deg}, "
            f"dwell={srch_cfg.dwell_time_s}s, "
            f"threshold={srch_cfg.no_detection_threshold_s}s"
        )
    else:
        print("PanTilt: DISABLED")

    # ------------------------ Run -------------------------
    TargetingProcessor(cam_cfg, det_cfg, trk_cfg, pt_cfg, srch_cfg).run()
    detach_usb_devices(_REQUIRED_BUSIDS)
    print("Main program finished.")


if __name__ == "__main__":
    main()

camera.py:
# camera.py
"""Thin VideoCapture wrapper with reconnection logic **and** V4L2 control
support (gain, contrast, sharpness, exposure …)."""

from __future__ import annotations

import math
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from face_tracking.config import CameraConfig


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None

        # Exposed runtime-queryable values
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0
        self.actual_fourcc_str: str = ""
        self.vertical_fov_deg: float = 0.0

    # ------------------------------------------------------------------ #
    #   I N T E R N A L   H E L P E R S
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_fourcc_str(fourcc_val: int) -> str:
        if fourcc_val == 0:
            return ""
        return "".join(chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4))

    def _calculate_vertical_fov(self) -> None:
        if (
            self.actual_width > 0
            and self.actual_height > 0
            and self.config.horizontal_fov_deg > 0
        ):
            hfov_rad = math.radians(self.config.horizontal_fov_deg)
            self.vertical_fov_deg = math.degrees(
                2
                * math.atan(
                    (self.actual_height / self.actual_width)
                    * math.tan(hfov_rad / 2.0)
                )
            )
        else:
            self.vertical_fov_deg = 0.0

    # ----------------  fallback to v4l2-ctl when OpenCV lacks a prop ----
    @staticmethod
    def _set_v4l2_ctrl(
        dev: int | str, name: str, value: int | float
    ) -> None:
        """
        Best-effort helper: silently ignore if v4l2-ctl is not installed.

        Parameters
        ----------
        dev   : int | str   /dev/videoX or numeric index passed to OpenCV
        name  : str         e.g. 'sharpness'
        value : int | float
        """
        node = f"/dev/video{dev}" if isinstance(dev, int) else str(dev)
        cmd = [
            "v4l2-ctl",
            "-d",
            node,
            "--set-ctrl",
            f"{name}={value}",
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
            )
            print(f"[Camera] v4l2-ctl set-ctrl {name}={value}")
        except FileNotFoundError:
            print("[Camera] Warning: v4l2-ctl not installed; skipping", name)
        except subprocess.CalledProcessError as exc:
            print(f"[Camera] v4l2-ctl error: {exc.stderr.decode().strip()}")

    # ------------------------------------------------------------------ #
    #   P U B L I C   A P I
    # ------------------------------------------------------------------ #
    def open(self) -> bool:
        """Open camera and apply resolution/fps **plus custom controls**."""
        # -------- open device -----------------------------------------
        backend = cv2.CAP_V4L2 if self.config.use_v4l2 else 0
        self.cap = cv2.VideoCapture(self.config.device_index, backend)
        if not self.cap or not self.cap.isOpened():
            print(f"[Camera] Could not open device {self.config.device_index}")
            self.cap = None
            return False

        # -------- core settings (res / fps / fourcc) ------------------
        if self.config.fourcc_str:
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc_str)
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps_request > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps_request)

        # -------- extended V4L2 controls ------------------------------
        # 1) exposure mode
        if self.config.auto_exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.config.auto_exposure)
        # 2) absolute exposure (must come *after* the mode)
        if self.config.exposure_time_absolute is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure_time_absolute)
        # 3) gain / contrast
        if self.config.gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)
        if self.config.contrast is not None:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
        # 4) sharpness – OpenCV added CAP_PROP_SHARPNESS in 4.5.4
        if self.config.sharpness is not None:
            if hasattr(cv2, "CAP_PROP_SHARPNESS"):
                self.cap.set(cv2.CAP_PROP_SHARPNESS, self.config.sharpness)
            else:
                self._set_v4l2_ctrl(
                    self.config.device_index,
                    "sharpness",
                    int(self.config.sharpness),
                )

        time.sleep(0.1)  # Let driver settle

        # -------- query what we actually got --------------------------
        self.actual_fourcc_str = self._get_fourcc_str(
            int(self.cap.get(cv2.CAP_PROP_FOURCC))
        )
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._calculate_vertical_fov()

        print(
            f"[Camera] {self.actual_width}x{self.actual_height}@{self.actual_fps:.1f} FPS "
            f"(FOURCC='{self.actual_fourcc_str}', VFOV={self.vertical_fov_deg:.1f} °)"
        )
        if self.actual_width == 0 or self.actual_height == 0:
            print("[Camera] Error: camera returned zero resolution")
            self.release()
            return False
        return True

    # ------------------------------------------------------------------ #
    #   S T A N D A R D   W R A P P E R S
    # ------------------------------------------------------------------ #
    def read(self) -> Tuple[float, Optional[np.ndarray]]:
        if not self.is_opened():
            return time.time(), None
        ts = time.time()
        ret, frame = self.cap.read()
        return (ts, frame) if ret and frame is not None else (ts, None)

    def is_opened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def release(self) -> None:
        if self.cap:
            print("[Camera] Releasing capture device")
            self.cap.release()
            self.cap = None

    # Convenience for other modules
    def get_properties(self) -> Tuple[int, int, float, str, float, float]:
        return (
            self.actual_width,
            self.actual_height,
            self.actual_fps,
            self.actual_fourcc_str,
            self.config.horizontal_fov_deg,
            self.vertical_fov_deg,
        )

common.py:
# common.py
"""Objects that are shared across multiple modules."""
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class TargetReport:
    """
    A single-frame snapshot of tracker state.
    All angles/positions are in *pixel* space; Pan-Tilt angles are handled elsewhere.
    """
    t_capture: float
    kf_now_px: Tuple[float, float]
    kf_future_px: Tuple[float, float]
    velocity_px_s: Tuple[float, float]
    bbox_px: Optional[Tuple[int, int, int, int]]
    img_size: Tuple[int, int]
    confidence: Optional[float]
    age_frames: int
    track_id: Optional[int]

config.py:
# config.py
"""Typed configuration blobs for the whole system."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps_request: int = 100
    use_v4l2: bool = True
    fourcc_str: str = "MJPG"
    horizontal_fov_deg: float = 70.0  # From camera data-sheet
    gain: int | None = 0              # 0‒100
    contrast: int | None = 32          # 0‒64
    sharpness: int | None = 3         # 0‒6
    auto_exposure: int | None = 3     # 1=manual, 3=auto
    exposure_time_absolute: int | None = 157  # 1‒5000 (1/10 000 s)


@dataclass
class DetectorConfig:
    model_selection: int = 0
    min_detection_confidence: float = 0.5
    min_bbox_size_for_tracking: int = 20


@dataclass
class TrackerConfig:
    # Lead time for future prediction (s)
    predict_lead_time_s: float = 0.05
    # Noise parameters tuned for pixel-level tracking
    measurement_noise_std: float = 3.0       # px
    process_noise_std: float = 70.0          # px/s²
    initial_velocity_error_std: float = 30.0  # px/s


@dataclass
class PanTiltConfig:
    port: Optional[str] = None
    baudrate: int = 250_000
    home_on_startup: bool = True
    rest_on_shutdown: bool = True
    kp_pan: float = 0.06
    kp_tilt: float = 0.03
    error_deadzone_px: int = 5
    min_angle_change_deg: float = 0.05
    max_cmd_rate_hz: float = 100.0
    use_blocking_moves: bool = False
    invert_pan_output: bool = False
    invert_tilt_output: bool = False


@dataclass
class SearchConfig:
    enabled: bool = True
    pan_angles_deg: List[float] = field(default_factory=lambda: [40.0, 0.0, -35.0, 0.0])
    tilt_deg: float = 0.0
    dwell_time_s: float = 1.0
    no_detection_threshold_s: float = 3.0

detector.py:
# detector.py
"""MediaPipe face-detection adapter."""
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import location_data_pb2

from face_tracking.config import DetectorConfig


class MediaPipeFaceDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=config.model_selection,
            min_detection_confidence=config.min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """
        Returns a list of ((x, y, w, h), confidence), with floats and clamped to image.
        """
        out: List[Tuple[Tuple[float, float, float, float], float]] = []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.detector.process(rgb)
        ih, iw = rgb.shape[:2]

        if results.detections:
            for det in results.detections:
                ld = det.location_data
                if not ld or ld.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                    continue
                bb = ld.relative_bounding_box
                x = bb.xmin * iw
                y = bb.ymin * ih
                w = bb.width * iw
                h = bb.height * ih
                # Clamp to valid region
                x = max(0.0, min(x, iw - w))
                y = max(0.0, min(y, ih - h))
                # Size threshold
                if w < self.config.min_bbox_size_for_tracking or h < self.config.min_bbox_size_for_tracking:
                    continue
                conf = float(det.score[0]) if det.score else 0.0
                out.append(((x, y, w, h), conf))

        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def close(self) -> None:
        self.detector.close()

helpers.py:
# helpers.py
"""Small utility classes that don’t fit elsewhere."""
from typing import Optional, Tuple

import numpy as np


class DetectionSmoother:
    """
    Exponential smoother for bounding-box center only.
    Width/height pass through unchanged.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.state_cx: Optional[float] = None
        self.state_cy: Optional[float] = None

    def update(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        bbox: (x, y, w, h)
        Returns (x, y, w, h) with x,y smoothed floats.
        """
        x, y, w, h = bbox
        raw_cx = x + w / 2.0
        raw_cy = y + h / 2.0

        if self.state_cx is None or self.state_cy is None:
            self.state_cx = raw_cx
            self.state_cy = raw_cy
        else:
            self.state_cx = self.alpha * raw_cx + (1.0 - self.alpha) * self.state_cx
            self.state_cy = self.alpha * raw_cy + (1.0 - self.alpha) * self.state_cy

        new_x = self.state_cx - w / 2.0
        new_y = self.state_cy - h / 2.0
        return (new_x, new_y, w, h)

live_tuning.py:
# live_tuning.py  (robust version)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


class RuntimeParamWatcher:
    """Watch a JSON file and hot-reload its contents when it changes."""

    def __init__(self, path: str | Path = "runtime_params.json") -> None:
        self.path = Path(path).expanduser().resolve()
        self._stamp: Tuple[float, int] = (0.0, -1)  # (mtime, size)
        self.params: Dict[str, Any] = {}

        print(f"[Runtime] Watching: {self.path}")
        self._load(initial=True)

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------
    def _load(self, *, initial: bool = False) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                self.params = json.load(fp)
            stat = self.path.stat()
            self._stamp = (stat.st_mtime, stat.st_size)
            if not initial:
                print(f"[Runtime] Reloaded parameters from {self.path}")
        except FileNotFoundError:
            if initial:
                print(
                    f"[Runtime] {self.path} not found – live-tuning disabled "
                    "(create the file to enable)."
                )
            else:
                print(f"[Runtime] {self.path} was deleted – keeping old params.")
        except json.JSONDecodeError as exc:
            print(f"[Runtime] JSON error in {self.path}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Runtime] Failed to reload {self.path}: {exc}")

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def maybe_reload(self) -> bool:
        """
        If the watched file changed since the last call reload it and
        return **True**, else return **False**.
        """
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False

        mtime, fsize = self._stamp
        # Some filesystems only update timestamps in 1- or 2-second ticks,
        # so we treat any change >=1 s *or* size change as “modified”.
        if stat.st_size != fsize or stat.st_mtime - mtime >= 1.0:
            self._load()
            return True
        return False

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.params.get(key, default)

pan_tilt.py:
# pan_tilt.py
"""Serial-controlled Pan-Tilt turret interface."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import serial


# ------------------- Exceptions / Enums -------------------
class FirmwareError(RuntimeError):
    """Raised when the firmware replies with an unexpected line."""


class _Ack(str, Enum):
    HOME_OK = auto()
    REST_OK = auto()
    MOVE_OK = auto()


_ACK_PATTERN = re.compile(r"^(HOME_OK|REST_OK|MOVE_OK)$")
_POS_STEPS   = re.compile(r"^(-?\d+),(-?\d+)$")
_POS_DEG     = re.compile(r"^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$")


# ------------------ Internal dataclass -------------------
@dataclass(slots=True)
class _SerialCfg:
    port: str
    baudrate: int = 250_000
    timeout: float = 2.0


# ---------------------- Main class ----------------------
class PanTilt:
    """High-level wrapper around the firmware’s ASCII protocol."""
    def __init__(
        self,
        port: str | Path,
        baudrate: int = 250_000,
        timeout: float = 10.0,
        write_timeout: float | None = None,
        *,
        eol: str = "\n",
        auto_flush: bool = True,
    ):
        self._cfg = _SerialCfg(str(port), baudrate, timeout)
        self._eol = eol.encode()
        self._auto_flush = auto_flush
        self._write_timeout = write_timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    # ---------------- Serial plumbing ----------------
    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return
        wt = self._write_timeout if self._write_timeout is not None else self._cfg.timeout
        self._ser = serial.Serial(
            port=self._cfg.port,
            baudrate=self._cfg.baudrate,
            timeout=self._cfg.timeout,
            write_timeout=wt,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.2)
        if self._ser.is_open:
            self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def is_open(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    # ------------------ Public API -------------------
    def home(self) -> None:
        self._cmd("HOME", expect=_Ack.HOME_OK)

    def rest(self) -> None:
        self._cmd("REST", expect=_Ack.REST_OK)

    def move_steps(self, pan: int, tilt: int) -> None:
        self._cmd(f"MOVEW {pan} {tilt}", expect=_Ack.MOVE_OK)

    def move_deg(self, pan_deg: float, tilt_deg: float) -> None:
        self._cmd(f"MOVE_DEGW {pan_deg:.3f} {tilt_deg:.3f}", expect=_Ack.MOVE_OK)

    def async_move_steps(self, pan: int, tilt: int) -> None:
        self._cmd(f"MOVE {pan} {tilt}", wait=False)

    def async_move_deg(self, pan_deg: float, tilt_deg: float) -> None:
        self._cmd(f"MOVE_DEG {pan_deg:.3f} {tilt_deg:.3f}", wait=False)

    def position_steps(self) -> Tuple[int, int]:
        line = self._cmd("POSITION", expect_pattern=_POS_STEPS)
        return tuple(map(int, line.split(",")))  # type: ignore[arg-type]

    def position_deg(self) -> Tuple[float, float]:
        line = self._cmd("POSITION_DEG", expect_pattern=_POS_DEG)
        return tuple(map(float, line.split(",")))  # type: ignore[arg-type]

    # -------------------- Tuning ---------------------
    def set_max_speed(self, value: float) -> None:
        self._cmd(f"SET MAX_SPEED {value}", wait=False)

    def set_max_accel(self, value: float) -> None:
        self._cmd(f"SET MAX_ACCEL {value}", wait=False)

    def set_rms_current(self, mA: int) -> None:
        self._cmd(f"SET RMS_CURRENT {mA}", wait=False)

    def set_stealth_chop(self, on: bool = True) -> None:
        self._cmd(f"SET STEALTHCHOP {'ON' if on else 'OFF'}", wait=False)

    # ----------------- Internal core -----------------
    def _cmd(
        self,
        cmd: str,
        expect: Optional[_Ack] = None,
        *,
        expect_pattern: Optional[re.Pattern[str]] = None,
        wait: bool = True,
    ) -> Optional[str]:
        if not self.is_open():
            raise RuntimeError("Serial port is not open")
        with self._lock:
            self._ser.write(cmd.upper().encode() + self._eol)
            if self._auto_flush:
                self._ser.flush()
            if not wait:
                return None
            while True:
                raw = self._ser.readline()
                if not raw:
                    raise FirmwareError(f"Timeout waiting for response to {cmd!r}")
                line = raw.decode(errors="replace").strip()
                if expect and line == expect.name:
                    return line
                if expect_pattern and expect_pattern.match(line):
                    return line

    # ---------------- Context / repr ---------------
    def __enter__(self) -> "PanTilt":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self.is_open() else "closed"
        return f"<PanTilt port={self._cfg.port!r} ({state})>"

processor.py:
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

tracker.py:
# tracker.py
"""4-state linear Kalman tracker with variable Δt **and live-tuning**."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from face_tracking.common import TargetReport
from face_tracking.config import TrackerConfig


class KalmanTracker:
    _id_counter = 0

    # ------------------------------------------------------------------ #
    #   I N I T
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: TrackerConfig, nominal_dt: float):
        self.cfg = cfg
        self.nominal_dt = nominal_dt

        # Build filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [
                [1, 0, nominal_dt, 0],
                [0, 1, 0, nominal_dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurement noise (R)
        mvar = cfg.measurement_noise_std**2
        self.kf.R = np.diag([mvar, mvar])

        # Process noise (Q)
        self._update_Q(nominal_dt)

        # Initial covariance
        pos_var = cfg.measurement_noise_std**2
        vel_var = cfg.initial_velocity_error_std**2
        self.kf.P = np.diag([pos_var, pos_var, vel_var, vel_var])

        # State
        self.kf.x = np.zeros((4, 1))
        self.initialized = False
        self.last_time: Optional[float] = None
        self.age_frames = 0
        self.track_id: Optional[int] = None
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_conf: Optional[float] = None

    # ------------------------------------------------------------------ #
    #   Q   B U I L D
    # ------------------------------------------------------------------ #
    def _update_Q(self, dt: float) -> None:
        """Rebuild Q for a given Δt and current process-noise σ."""
        pvar = self.cfg.process_noise_std**2
        self.kf.Q = Q_discrete_white_noise(
            dim=2, dt=dt, var=pvar, order_by_dim=False, block_size=2
        )

    # ------------------------------------------------------------------ #
    #   S T A T I C   ID
    # ------------------------------------------------------------------ #
    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    # ------------------------------------------------------------------ #
    #   N O M I N A L   Δt  U P D A T E
    # ------------------------------------------------------------------ #
    def update_nominal_dt(self, dt: float) -> None:
        """If camera FPS changed, adjust F and Q."""
        if abs(self.nominal_dt - dt) > 1e-6:
            self.nominal_dt = dt
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self._update_Q(dt)

    # ------------------------------------------------------------------ #
    #   L I V E   T U N I N G   A P I
    # ------------------------------------------------------------------ #
    def apply_tuning(
        self,
        *,
        predict_lead_time_s: float | None = None,
        measurement_noise_std: float | None = None,
        process_noise_std: float | None = None,
    ) -> None:
        """
        Dynamically update key parameters while running.
        Only rebuilds matrices that actually changed.
        """
        if predict_lead_time_s is not None:
            self.cfg.predict_lead_time_s = float(predict_lead_time_s)

        if measurement_noise_std is not None and measurement_noise_std > 0:
            self.cfg.measurement_noise_std = float(measurement_noise_std)
            mvar = self.cfg.measurement_noise_std**2
            self.kf.R = np.diag([mvar, mvar])

        if process_noise_std is not None and process_noise_std > 0:
            self.cfg.process_noise_std = float(process_noise_std)
            self._update_Q(self.nominal_dt)

    # ------------------------------------------------------------------ #
    #   P R E D I C T   +   U P D A T E
    # ------------------------------------------------------------------ #
    def predict_and_update(
        self,
        measurement: Optional[Tuple[float, float]],
        timestamp: float,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        # ----- Δt clamping + predict -----
        dt = self.nominal_dt
        if self.last_time is not None:
            actual_dt = timestamp - self.last_time
            if actual_dt > 1e-6:
                min_dt = 0.5 * self.nominal_dt
                max_dt = 2.0 * self.nominal_dt
                dt = float(np.clip(actual_dt, min_dt, max_dt))
                self.kf.F[0, 2] = dt
                self.kf.F[1, 3] = dt
                self._update_Q(dt)
        self.kf.predict()

        # ----- conditional update -----
        if measurement is not None:
            z = np.array([[measurement[0]], [measurement[1]]])
            if not self.initialized:
                # First detection initializes state.
                self.kf.x[0, 0] = measurement[0]
                self.kf.x[1, 0] = measurement[1]
                self.kf.x[2, 0] = 0.0
                self.kf.x[3, 0] = 0.0
                # Reset covariance
                pos_var = self.cfg.measurement_noise_std**2
                vel_var = self.cfg.initial_velocity_error_std**2
                self.kf.P = np.diag([pos_var, pos_var, vel_var, vel_var])
                self.track_id = self._next_id()
                self.initialized = True
                self.age_frames = 1
            else:
                self.kf.update(z)
                self.age_frames += 1

            self.last_bbox = bbox
            self.last_conf = confidence
        else:
            # Lost detection
            self.age_frames = 0
            self.last_bbox = None
            self.last_conf = None

        self.last_time = timestamp

    # ------------------------------------------------------------------ #
    #   R E P O R T
    # ------------------------------------------------------------------ #
    def get_report(self) -> Optional[TargetReport]:
        if not self.initialized or self.last_time is None:
            return None
        x, y, vx, vy = self.kf.x.flatten()
        fx = x + vx * self.cfg.predict_lead_time_s
        fy = y + vy * self.cfg.predict_lead_time_s
        return TargetReport(
            t_capture=self.last_time,
            kf_now_px=(x, y),
            kf_future_px=(fx, fy),
            velocity_px_s=(vx, vy),
            bbox_px=self.last_bbox,
            img_size=(0, 0),  # filled by processor
            confidence=self.last_conf,
            age_frames=self.age_frames,
            track_id=self.track_id,
        )

stepper_driver.ino (This script is ran on the Arduino Nano ESP32 for controlling the TMC2209 drivers / steppers:
// ──────────────────────────────────────────────────────────────────────────────
//  Pan-Tilt Turret Firmware   (Arduino Nano ESP32 | TMC2209 | AccelStepper)
//  v2025-06-22 r  – 4-µstep edition    (“r” = REST-speed fix)
//      • Added speed-aware coordinatedMove() overload
//      • goRest() now travels at HOMING_SPEED_SLOW (same as home creep)
//      • Removed unused REST_SPEED constant
// ──────────────────────────────────────────────────────────────────────────────

#include <Arduino.h>
#include <TMCStepper.h>
#include <AccelStepper.h>
#include <math.h>

/* ───── Mechanics & gearing ───── */
constexpr int     FULL_STEPS_PER_REV = 200;
constexpr uint8_t MICROSTEPS         = 8;            // controller resolution
constexpr float   PAN_GEAR_RATIO     = 80.0f / 20.0f; // 4 : 1
constexpr float   TILT_GEAR_RATIO    = 60.0f / 20.0f; // 3 : 1

/*  Steps-per-rev **at the axis** (µsteps)                         */
constexpr float SPR_PAN  = FULL_STEPS_PER_REV * MICROSTEPS * PAN_GEAR_RATIO;  // 3 200
constexpr float SPR_TILT = FULL_STEPS_PER_REV * MICROSTEPS * TILT_GEAR_RATIO; // 2 400

/* ───── Reference offsets ───── */
constexpr long PAN_ZERO_BIAS_STEPS  =  1550;   // centre reference (+1550 /2)
constexpr long TILT_ZERO_BIAS_STEPS =  525;   // (+525 /2)

/* ───── Software travel limits (µsteps) ───── */
constexpr long PAN_MIN_POS  = 0;
constexpr long PAN_MAX_POS  = 4800;   // ≈ 270 °
constexpr long TILT_MIN_POS = 0;
constexpr long TILT_MAX_POS =  1100;   // ≈ 82.5 °

/* ───── I/O pins ───── */
constexpr uint8_t EN_PIN_PAN    = 2;
constexpr uint8_t STEP_PIN_PAN  = 3;
constexpr uint8_t DIR_PIN_PAN   = 4;
constexpr uint8_t EN_PIN_TILT   = 5;
constexpr uint8_t STEP_PIN_TILT = 6;
constexpr uint8_t DIR_PIN_TILT  = 7;
constexpr uint8_t LIMIT_PIN_PAN  = A6;
constexpr uint8_t LIMIT_PIN_TILT = A7;
constexpr uint8_t LIMIT_TRIGGERED = LOW;

/* ───── Homing offsets (pull-off distance after switch) ───── */
constexpr long PAN_OFFSET_FROM_MIN  =  100;  // 100/2
constexpr long TILT_OFFSET_FROM_MIN =  150;  // 150/2

/* ───── REST pose ───── */
constexpr long REST_PAN_POSITION  = PAN_ZERO_BIAS_STEPS;
constexpr long REST_TILT_POSITION = TILT_ZERO_BIAS_STEPS;

/* ───── TMC2209 drivers ───── */
constexpr float   R_SENSE         = 0.11f;
constexpr uint8_t DRIVER_ADDR_PAN  = 0b00;
constexpr uint8_t DRIVER_ADDR_TILT = 0b01;
#define SERIAL_PORT_TMC Serial0

TMC2209Stepper driver_pan (&SERIAL_PORT_TMC, R_SENSE, DRIVER_ADDR_PAN );
TMC2209Stepper driver_tilt(&SERIAL_PORT_TMC, R_SENSE, DRIVER_ADDR_TILT);

/* ───── Motion defaults (µsteps s⁻¹ and µsteps s⁻²) ───── */
float    maxSpeed    = 75000.0f;   // 50 000 /2
float    maxAccel    = 100000.0f;   // 50 000 /2
uint16_t rmsCurrent  = 900;        // mA
bool     stealthChop = true;

/* ───── Misc speeds ───── */
constexpr float HOMING_SPEED_FAST = 500.0f;
constexpr float REST_SPEED =        2000.0f;

constexpr int   BACKOFF_STEPS     = 200;     // 200/2

/* ───── Stepper instances ───── */
AccelStepper stepper_pan (AccelStepper::DRIVER, STEP_PIN_PAN , DIR_PIN_PAN );
AccelStepper stepper_tilt(AccelStepper::DRIVER, STEP_PIN_TILT, DIR_PIN_TILT);

/* ───── Helper functions ───── */
static inline long clampPan (long s){ return s < PAN_MIN_POS  ? PAN_MIN_POS  : (s > PAN_MAX_POS  ? PAN_MAX_POS  : s); }
static inline long clampTilt(long s){ return s < TILT_MIN_POS ? TILT_MIN_POS : (s > TILT_MAX_POS ? TILT_MAX_POS : s); }

inline long  panDegToSteps (float d){ return clampPan ( REST_PAN_POSITION  + lroundf(d * SPR_PAN  / 360.0f) ); }
inline long  tiltDegToSteps(float d){ return clampTilt( REST_TILT_POSITION + lroundf(d * SPR_TILT / 360.0f) ); }
inline float panStepsToDeg (long s ){ return (s - REST_PAN_POSITION ) * 360.0f / SPR_PAN; }
inline float tiltStepsToDeg(long s ){ return (s - REST_TILT_POSITION) * 360.0f / SPR_TILT; }

/* ───── Driver & stepper configuration helpers ───── */
void cfgDriver(TMC2209Stepper& d){
  d.rms_current(rmsCurrent);
  d.microsteps(MICROSTEPS);
  d.en_spreadCycle(!stealthChop);
}
void cfgStepper(AccelStepper& s,uint8_t enPin){
  s.setMaxSpeed(maxSpeed);
  s.setAcceleration(maxAccel);
  s.setEnablePin(enPin);
  s.setPinsInverted(false,false,true);
  pinMode(enPin,OUTPUT);
  digitalWrite(enPin,LOW);
  s.enableOutputs();
}
void applyAllConfigs(){
  for(auto drv : {&driver_pan,&driver_tilt}){
    drv->begin();
    drv->toff(4);
    drv->blank_time(24);
    drv->pdn_disable(true);
    drv->I_scale_analog(false);
    drv->GCONF(drv->GCONF() | (1<<11));          // internal CLK
    drv->pwm_autoscale(true);
  }
  cfgDriver(driver_pan);
  cfgDriver(driver_tilt);
  cfgStepper(stepper_pan ,EN_PIN_PAN );
  cfgStepper(stepper_tilt,EN_PIN_TILT);
}

/* ───── Coordinated motion helpers ───── */
/*  Variant with explicit baseSpeed (µsteps s⁻¹) */
void coordinatedMove(long p,long t,float baseSpeed){
  p = clampPan(p);
  t = clampTilt(t);
  long dp = labs(p - stepper_pan.currentPosition());
  long dt = labs(t - stepper_tilt.currentPosition());
  if(!dp && !dt) return;

  const float rp = dp / SPR_PAN;   // travel in revs
  const float rt = dt / SPR_TILT;
  float sp = baseSpeed, st = baseSpeed;
  float ap = maxAccel,  at = maxAccel;

  if(rp > rt && rp > 0){
    float k = rt / rp; st *= k; at *= k;
  }else if(rt > rp && rt > 0){
    float k = rp / rt; sp *= k; ap *= k;
  }

  stepper_pan.setMaxSpeed(sp);   stepper_pan.setAcceleration(ap);
  stepper_tilt.setMaxSpeed(st);  stepper_tilt.setAcceleration(at);

  stepper_pan.moveTo(p);
  stepper_tilt.moveTo(t);
}
/*  Back-compatibility wrapper (uses global maxSpeed exactly as before) */
inline void coordinatedMove(long p,long t){
  coordinatedMove(p,t,maxSpeed);
}

/* — blocking variant, prints MOVE_OK — */
void moveAndWait(long p,long t){
  coordinatedMove(p,t);
  while(stepper_pan.distanceToGo() || stepper_tilt.distanceToGo()){
    stepper_pan.run();
    stepper_tilt.run();
  }
  Serial.println(F("MOVE_OK"));
}
void moveDegAndWait(float pd,float td){
  moveAndWait(panDegToSteps(pd), tiltDegToSteps(td));
}

/* ───── Homing & REST ───── */
void homeAxis(AccelStepper& s,uint8_t limit,long off,bool rev=false){
  s.setAcceleration(maxAccel/2);
  s.setMaxSpeed(HOMING_SPEED_FAST);
  s.setSpeed(rev ?  HOMING_SPEED_FAST : -HOMING_SPEED_FAST);
  while(digitalRead(limit) != LIMIT_TRIGGERED) s.runSpeed();

  s.setSpeed(rev ? -HOMING_SPEED_FAST : HOMING_SPEED_FAST);
  for(int i=0;i<BACKOFF_STEPS;++i) s.runSpeed();

  s.setSpeed(rev ?  HOMING_SPEED_FAST : -HOMING_SPEED_FAST);
  while(digitalRead(limit) != LIMIT_TRIGGERED) s.runSpeed();

  s.setSpeed(rev ? -HOMING_SPEED_FAST : HOMING_SPEED_FAST);
  for(int i=0;i<BACKOFF_STEPS;++i) s.runSpeed();

  s.setCurrentPosition(-off);          // zero reference
  s.setMaxSpeed(HOMING_SPEED_FAST);
  s.moveTo(0);
  while(s.run()){}
  s.setMaxSpeed(maxSpeed);
  s.setAcceleration(maxAccel);
}
void doHome(bool ack=true){
  homeAxis(stepper_tilt, LIMIT_PIN_TILT, TILT_OFFSET_FROM_MIN);
  homeAxis(stepper_pan , LIMIT_PIN_PAN , PAN_OFFSET_FROM_MIN );
  if(ack) Serial.println(F("HOME_OK"));
}
void goRest(){
  /* travel at the same gentle speed used for the final homing creep */
  coordinatedMove(REST_PAN_POSITION, REST_TILT_POSITION, REST_SPEED);
  while(stepper_pan.distanceToGo() || stepper_tilt.distanceToGo()){
    stepper_pan.run();
    stepper_tilt.run();
  }

  stepper_pan.setMaxSpeed(maxSpeed);   stepper_pan.setAcceleration(maxAccel);
  stepper_tilt.setMaxSpeed(maxSpeed);  stepper_tilt.setAcceleration(maxAccel);

  Serial.println(F("REST_OK"));
}

/* ───── SET helper ───── */
void doSet(const char* n,const char* v){
  if(!strcmp(n,"MAX_SPEED")){
    maxSpeed = atof(v);
    stepper_pan.setMaxSpeed(maxSpeed);
    stepper_tilt.setMaxSpeed(maxSpeed);
  }else if(!strcmp(n,"MAX_ACCEL")){
    maxAccel = atof(v);
    stepper_pan.setAcceleration(maxAccel);
    stepper_tilt.setAcceleration(maxAccel);
  }else if(!strcmp(n,"RMS_CURRENT")){
    rmsCurrent = atoi(v);
    driver_pan.rms_current(rmsCurrent);
    driver_tilt.rms_current(rmsCurrent);
  }else if(!strcmp(n,"STEALTHCHOP")){
    stealthChop = (strcasecmp(v,"ON")==0 || !strcmp(v,"1"));
    driver_pan.en_spreadCycle(!stealthChop);
    driver_tilt.en_spreadCycle(!stealthChop);
  }
}

/* ───── Position reporters ───── */
void sendPosSteps(){
  Serial.print(stepper_pan.currentPosition());
  Serial.print(',');
  Serial.println(stepper_tilt.currentPosition());
}
void sendPosDeg(){
  Serial.print(panStepsToDeg(stepper_pan.currentPosition()),2);
  Serial.print(',');
  Serial.println(tiltStepsToDeg(stepper_tilt.currentPosition()),2);
}

/* ───── Arduino setup / loop ───── */
void setup(){
  Serial.begin(250000);
  SERIAL_PORT_TMC.begin(250000);

  pinMode(LIMIT_PIN_PAN , INPUT_PULLUP);
  pinMode(LIMIT_PIN_TILT, INPUT_PULLUP);

  applyAllConfigs();
  doHome(true);
  goRest();
}

void loop(){
  stepper_pan.run();
  stepper_tilt.run();

  if(!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if(line.isEmpty()) return;
  line.toUpperCase();

  /* ───── Command parser (specific → generic) ───── */
  if     (line == "HOME")         doHome(true);
  else if(line == "REST")         goRest();
  else if(line == "POSITION")     sendPosSteps();
  else if(line == "POSITION_DEG") sendPosDeg();

  /* degree-based */
  else if(line.startsWith("MOVE_DEGW ")){
    float pd, td;
    if(sscanf(line.c_str()+9,"%f %f",&pd,&td) == 2) moveDegAndWait(pd, td);
  }
  else if(line.startsWith("MOVE_DEG ")){
    float pd, td;
    if(sscanf(line.c_str()+8,"%f %f",&pd,&td) == 2)
      coordinatedMove(panDegToSteps(pd), tiltDegToSteps(td));
  }

  /* raw micro-step commands */
  else if(line.startsWith("MOVEW ")){
    long p, t;
    if(sscanf(line.c_str()+6,"%ld %ld",&p,&t) == 2) moveAndWait(p, t);
  }
  else if(line.startsWith("MOVE ")){
    long p, t;
    if(sscanf(line.c_str()+5,"%ld %ld",&p,&t) == 2) coordinatedMove(p, t);
  }

  /* settings */
  else if(line.startsWith("SET ")){
    char n[16], v[16];
    if(sscanf(line.c_str()+4,"%15s %15s", n, v) == 2) doSet(n, v);
  }
}
