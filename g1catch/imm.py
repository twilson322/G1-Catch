"""Ball tracker: an interacting-multiple-model pair of UKFs plus gating, timing and forecasting.

Mode 0 (flight) trusts the ballistic model. Mode 1 (impact) allows large velocity changes and
takes over for a few frames at a bounce or any other event the flight model cannot explain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from . import ballistics as bal
from .ukf import Measurement, Ukf, propagate, rollout, sigma_points
from .vision import Intrinsics, project, triangulate, triangulation_covariance


MAX_SANE_VARIANCE = 1e6      # m^2 or (m/s)^2: beyond this the estimate carries no information


@dataclass
class TrackerConfig:
    pixel_sigma: float = 0.5                       # replaced by the detector's measured error
    flight_accel_noise: float = 0.3                # (m/s^2)^2 s, unmodelled drag and spin
    impact_accel_noise: float = 600.0              # lets velocity jump by metres per second per frame
    # Row i gives the probability of moving from mode i to each mode in one camera frame. An
    # impact needs something to hit, so the chance of entering the impact mode depends on height:
    # the first matrix applies within near_floor_m of the floor, the second in open air. With one
    # matrix for both, a ball that leaves the cameras' view for a few frames has its covariance
    # inflated by an impact that cannot have happened.
    mode_transition: np.ndarray = field(default_factory=lambda: np.array([[0.90, 0.10], [0.60, 0.40]]))
    mode_transition_open_air: np.ndarray = field(default_factory=lambda: np.array([[0.998, 0.002], [0.60, 0.40]]))
    near_floor_m: float = 0.25
    initial_mode_probability: tuple = (0.9, 0.1)
    gate_probability: float = 0.999                # chi-squared gate on the innovation, 1.0 disables it
    # Velocity prior at initialisation: a throw toward the robot. Wide enough to be uninformative
    # after three frames.
    initial_velocity: tuple = (-5.0, 0.0, 1.0)
    initial_velocity_sigma: tuple = (4.0, 3.0, 4.0)
    use_imm: bool = True
    # A track may only start inside this box (world frame, metres). It covers the space a throw
    # can come from and rejects detections of things on the far wall.
    start_volume_lo: tuple = (0.3, -2.5, 0.0)
    start_volume_hi: tuple = (5.5, 2.5, 3.5)
    max_pairing_error_px: float = 2.0              # reprojection error allowed when pairing detections


class BallTracker:
    def __init__(self, intr: Intrinsics, cfg: TrackerConfig | None = None, kd: float = bal.NOMINAL_KD,
                 residual=None):
        self.cfg = cfg or TrackerConfig()
        self.intr = intr
        self.kd = kd
        self.residual = residual
        noises = (self.cfg.flight_accel_noise, self.cfg.impact_accel_noise)
        if not self.cfg.use_imm:
            noises = noises[:1]
        self.filters = [Ukf(intr, q, self.cfg.pixel_sigma, kd, residual) for q in noises]
        self.mu = np.array(self.cfg.initial_mode_probability[:len(self.filters)], dtype=float)
        self.mu /= self.mu.sum()
        self.time = 0.0
        self.initialised = False
        self.updates = 0
        self.last_nis = float("nan")

    # --- lifecycle ---------------------------------------------------------------------------
    def initialise(self, meas: Measurement, t: float) -> bool:
        """Start a track from one frame that both cameras saw."""
        p = self._starting_point(meas)
        if p is None:
            return False
        cov_p = triangulation_covariance(self.intr, meas.poses, p, self.cfg.pixel_sigma)
        x = np.concatenate([p, self.cfg.initial_velocity])
        cov = np.zeros((6, 6))
        cov[:3, :3] = cov_p
        cov[3:, 3:] = np.diag(np.square(self.cfg.initial_velocity_sigma))
        for f in self.filters:
            f.x, f.P = x.copy(), cov.copy()
        self.time = t
        self.initialised = True
        self.updates = 1
        return True

    def _starting_point(self, meas: Measurement) -> np.ndarray | None:
        """Triangulated position of the ball for a new track, or None if no pair of detections
        is consistent with a ball inside the start volume.

        With several detections per image, every left-right pair is tried. A wrong pairing shows
        up as a point outside the volume or as rays that do not meet (reprojection error).
        """
        if meas.candidates is not None:
            left, right = meas.candidates
        else:
            left, right = ([px] if px is not None else [] for px in meas.pixels)
        best, best_error = None, self.cfg.max_pairing_error_px
        for pl in left:
            for pr in right:
                point = triangulate(self.intr, meas.poses, (pl, pr))
                if np.any(point < self.cfg.start_volume_lo) or np.any(point > self.cfg.start_volume_hi):
                    continue
                error = max(np.linalg.norm(project(self.intr, pose, point)[0] - px)
                            for pose, px in zip(meas.poses, (pl, pr)))
                if error < best_error:
                    best, best_error = (point, pl, pr), error
        if best is None:
            return None
        meas.pixels = (best[1], best[2])
        return best[0]

    def step(self, meas: Measurement | None, t: float) -> bool:
        """Advance to capture time t and apply the frame's measurement. Returns True if it was used."""
        try:
            used = self._step(meas, t)
        except np.linalg.LinAlgError:
            return self._lose_track()
        if not all(np.all(np.isfinite(f.x)) and np.all(np.isfinite(f.P)) and np.abs(f.P).max() < MAX_SANE_VARIANCE
                   for f in self.filters):
            return self._lose_track()
        return used

    def _lose_track(self) -> bool:
        """The estimate has become numerically meaningless. With the gate on this does not happen.
        With it off, one false detection can move the estimate to where the projection is
        degenerate (behind a camera). The caller starts a new track."""
        self.initialised = False
        return False

    def _step(self, meas: Measurement | None, t: float) -> bool:
        dt = t - self.time
        self.time = t
        if len(self.filters) == 1:
            f = self.filters[0]
            f.predict(dt)
            if meas is None or not meas.cameras() or not self._gate(meas):
                return False
            self.last_nis, _ = f.update(meas)
            self.updates += 1
            return True

        # 1. Mixing: each filter restarts from a blend of all filters' estimates.
        height = self.mean[2] - bal.BALL_RADIUS_M
        pi = self.cfg.mode_transition if height < self.cfg.near_floor_m else self.cfg.mode_transition_open_air
        c = pi.T @ self.mu                                   # predicted mode probabilities
        w = (pi * self.mu[:, None]) / c[None, :]             # w[i, j] = P(was i | now j)
        xs = [f.x.copy() for f in self.filters]
        ps = [f.P.copy() for f in self.filters]
        for j, f in enumerate(self.filters):
            x0 = sum(w[i, j] * xs[i] for i in range(2))
            p0 = sum(w[i, j] * (ps[i] + np.outer(xs[i] - x0, xs[i] - x0)) for i in range(2))
            f.x, f.P = x0, p0
            f.predict(dt)
        self.mu = c
        if meas is None or not meas.cameras() or not self._gate(meas):
            return False
        # 2. Update each filter and reweight the modes by how well each predicted the frame.
        nis_by_mode = np.empty(2)
        likelihood = np.empty(2)
        try:
            for j, f in enumerate(self.filters):
                nis_by_mode[j], likelihood[j] = f.update(meas)
        except np.linalg.LinAlgError:
            return self._lose_track()
        post = likelihood * c
        self.mu = post / post.sum() if post.sum() > 0 else c
        self.last_nis = float(self.mu @ nis_by_mode)
        self.updates += 1
        return True

    def _gate(self, meas: Measurement) -> bool:
        """Reject a frame that even the most permissive mode finds implausible."""
        if self.cfg.gate_probability >= 1.0:
            return True                                    # gate disabled
        dim = 2 * len(meas.cameras())
        limit = chi2.ppf(self.cfg.gate_probability, dim)
        best = np.inf
        for f in self.filters:
            nu, s, _ = f.innovation(meas)
            best = min(best, float(nu @ np.linalg.solve(s, nu)))
        return best < limit

    # --- outputs -----------------------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        return sum(m * f.x for m, f in zip(self.mu, self.filters))

    @property
    def covariance(self) -> np.ndarray:
        x = self.mean
        return sum(m * (f.P + np.outer(f.x - x, f.x - x)) for m, f in zip(self.mu, self.filters))

    def forecast(self, t_start: float, dt: float, steps: int):
        """Ballistic forecast from the current estimate.

        Returns times (steps,), mean positions and velocities (steps, 3) and position covariances
        (steps, 3, 3) at t_start + dt * (1..steps). t_start may be later than the last
        measurement, which is how camera latency is compensated.
        """
        pts, w = sigma_points(self.mean, self.covariance)
        lead = t_start - self.time
        if lead > 0:
            pts = propagate(pts, lead, self.kd, self.residual)
        states = rollout(pts, dt, steps, self.kd, self.residual)   # substeps internally, so dt may be long
        ps, vs = states[:, :, :3], states[:, :, 3:]
        mean_p = np.einsum("s,ksi->ki", w, ps)
        mean_v = np.einsum("s,ksi->ki", w, vs)
        dev = ps - mean_p[:, None, :]
        cov = np.einsum("s,ksi,ksj->kij", w, dev, dev)
        # Process noise accumulated over the forecast, position block only.
        horizon = lead + dt * np.arange(1, steps + 1)
        cov += (self.cfg.flight_accel_noise * horizon**3 / 3.0)[:, None, None] * np.eye(3)
        return t_start + dt * np.arange(1, steps + 1), mean_p, mean_v, cov
