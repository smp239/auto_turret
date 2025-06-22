# config.py
"""Typed configuration blobs for the whole system."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------- Camera ----------------------
@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps_request: int = 100
    use_v4l2: bool = True
    fourcc_str: str = "MJPG"
    horizontal_fov_deg: float = 70.0  # From camera data-sheet


# --------------------- Detector ---------------------
@dataclass
class DetectorConfig:
    model_selection: int = 0
    min_detection_confidence: float = 0.25
    min_bbox_size_for_tracking: int = 30


# ---------------------- Tracker ---------------------
@dataclass
class TrackerConfig:
    predict_lead_time_s: float = 0.15
    measurement_noise_std: float = 10.0
    process_noise_std: float = 150.0
    initial_velocity_error_std: float = 100.0


# ---------------------- PanTilt ---------------------
@dataclass
class PanTiltConfig:
    port: Optional[str] = None          # Set to "/dev/ttyACM0" or COM-port to enable
    baudrate: int = 250_000
    home_on_startup: bool = True
    rest_on_shutdown: bool = True
    kp_pan: float = 0.08
    kp_tilt: float = 0.06
    error_deadzone_px: int = 15
    min_angle_change_deg: float = 0.05
    max_cmd_rate_hz: float = 100.0
    use_blocking_moves: bool = False
    invert_pan_output: bool = False
    invert_tilt_output: bool = False


# ----------------------- Search ---------------------
@dataclass
class SearchConfig:
    enabled: bool = True
    pan_angles_deg: List[float] = field(default_factory=lambda: [40.0, 0.0, -35.0, 0.0])
    tilt_deg: float = 0.0
    dwell_time_s: float = 1.0
    no_detection_threshold_s: float = 3.0



