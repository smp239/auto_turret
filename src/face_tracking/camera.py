# camera.py
"""A thin wrapper around cv2.VideoCapture with reconnection logic."""
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig


class Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None

        # Exposed runtime values
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_fourcc_str = ""
        self.vertical_fov_deg = 0.0

    # --------------- Internal helpers ---------------
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
                2 * math.atan(
                    (self.actual_height / self.actual_width) * math.tan(hfov_rad / 2.0)
                )
            )
        else:
            self.vertical_fov_deg = 0.0

    # --------------- Public API ---------------------
    def open(self) -> bool:
        # Choose backend
        if self.config.use_v4l2:
            self.cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.config.device_index)

        if not self.cap or not self.cap.isOpened():
            print(f"[Camera] Could not open device {self.config.device_index}")
            self.cap = None
            return False

        # Apply settings
        if self.config.fourcc_str:
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc_str)
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps_request > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps_request)

        time.sleep(0.1)  # Let driver settle

        # Query what we actually got
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
