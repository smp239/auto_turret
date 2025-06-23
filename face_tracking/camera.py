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
