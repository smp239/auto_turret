# src/face_tracking/__init__.py
"""Face-tracking package – re-export high-level API."""
from .processor import TargetingProcessor        # noqa: F401
from .config import (                            # noqa: F401
    CameraConfig, DetectorConfig, TrackerConfig,
    PanTiltConfig, SearchConfig,
)
