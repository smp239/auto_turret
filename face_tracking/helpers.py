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
