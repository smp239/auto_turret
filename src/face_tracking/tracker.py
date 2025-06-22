# tracker.py
"""4-state linear Kalman tracker with variable Δt."""
from typing import Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from common import TargetReport
from config import TrackerConfig


class KalmanTracker:
    _id_counter = 0

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

        mvar = cfg.measurement_noise_std ** 2
        self.kf.R = np.diag([mvar, mvar])

        self._update_Q(nominal_dt)

        self.kf.P = np.eye(4) * 500.0
        self.kf.x = np.zeros((4, 1))

        # Runtime bookkeeping
        self.initialized = False
        self.last_time: Optional[float] = None
        self.age_frames = 0
        self.track_id: Optional[int] = None
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_conf: Optional[float] = None

    # ----------------- Private helpers -----------------
    def _update_Q(self, dt: float) -> None:
        pvar = self.cfg.process_noise_std ** 2
        self.kf.Q = Q_discrete_white_noise(
            dim=2, dt=dt, var=pvar, order_by_dim=False, block_size=2
        )

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    # ------------------ Public API --------------------
    def update_nominal_dt(self, dt: float) -> None:
        if abs(self.nominal_dt - dt) > 1e-6:
            self.nominal_dt = dt
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self._update_Q(dt)

    def predict_and_update(
        self,
        measurement: Optional[Tuple[float, float]],
        timestamp: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        # Predict step – adapt Δt if capture jittered
        dt = self.nominal_dt
        if self.last_time is not None:
            actual_dt = timestamp - self.last_time
            if actual_dt > 1e-6:
                dt = actual_dt
                self.kf.F[0, 2] = dt
                self.kf.F[1, 3] = dt
                self._update_Q(dt)
        self.kf.predict()

        # Update step
        if measurement is not None:
            z = np.array([[measurement[0]], [measurement[1]]])
            if not self.initialized:
                # First detection → initialise state
                self.kf.x[0, 0], self.kf.x[1, 0] = measurement
                self.kf.x[2, 0] = self.kf.x[3, 0] = 0.0

                pos_var = self.cfg.measurement_noise_std ** 2
                vel_var = self.cfg.initial_velocity_error_std ** 2
                self.kf.P = np.diag([pos_var, pos_var, vel_var, vel_var])

                self.track_id = self._next_id()
                self.initialized = True
                self.age_frames = 1
            else:
                self.age_frames += 1

            self.kf.update(z)
            self.last_bbox = bbox
            self.last_conf = confidence
        else:
            # Lost target
            self.age_frames = 0
            self.last_bbox = None
            self.last_conf = None

        self.last_time = timestamp

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
            img_size=(0, 0),          # Processor fills this in
            confidence=self.last_conf,
            age_frames=self.age_frames,
            track_id=self.track_id,
        )
