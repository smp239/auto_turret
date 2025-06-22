# helpers.py
"""Small utility classes that don’t fit elsewhere."""
from typing import Optional, Tuple

import numpy as np


class DetectionSmoother:
    """
    Simple exponential smoother for bounding-box (x, y, w, h) tuples.
    Helps stabilize the raw detector before Kalman filtering.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        meas = np.array(bbox, dtype=float)
        if self.state is None:
            self.state = meas.copy()
        else:
            self.state = self.alpha * meas + (1.0 - self.alpha) * self.state
        return tuple(self.state.astype(int))
