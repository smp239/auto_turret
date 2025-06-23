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
