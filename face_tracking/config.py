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
