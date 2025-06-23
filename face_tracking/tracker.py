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
