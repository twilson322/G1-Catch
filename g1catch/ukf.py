"""Unscented Kalman filter for the ball state [position, velocity] with stereo pixel measurements."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ballistics as bal
from .vision import CameraPose, Intrinsics, project

STATE_DIM = 6
MAX_SUBSTEP_S = 0.004          # RK4 substep inside a predict
SIGMA_LAMBDA = 1.0             # centre weight lambda / (n + lambda) = 1/7


MIN_COVARIANCE_EIGENVALUE = 1e-10


def symmetric_positive(cov: np.ndarray) -> np.ndarray:
    """`cov` made symmetric, with any eigenvalue below a small floor raised to it.

    The update P - K S K' is a difference of two positive matrices. After a measurement far out
    in the tail (a false detection that passed the gate) round-off can leave it slightly
    indefinite, and the next Cholesky factorisation would fail.
    """
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    if values[0] >= MIN_COVARIANCE_EIGENVALUE:
        return cov
    return (vectors * np.maximum(values, MIN_COVARIANCE_EIGENVALUE)) @ vectors.T


def sigma_points(mean: np.ndarray, cov: np.ndarray):
    """2n+1 points and weights that reproduce `mean` and `cov` exactly."""
    n = mean.size
    scale = n + SIGMA_LAMBDA
    root = np.linalg.cholesky(scale * symmetric_positive(cov))
    pts = np.vstack([mean, mean + root.T, mean - root.T])
    w = np.full(2 * n + 1, 0.5 / scale)
    w[0] = SIGMA_LAMBDA / scale
    return pts, w


def white_acceleration_noise(q: float, dt: float) -> np.ndarray:
    """Discrete process noise for acceleration noise of spectral density q ((m/s^2)^2 s)."""
    i3 = np.eye(3)
    return q * np.block([[dt**3 / 3 * i3, dt**2 / 2 * i3], [dt**2 / 2 * i3, dt * i3]])


try:                                            # compiled flight model, see cpp/flight_core.cpp
    from . import _flight_core
except ImportError:                             # not built: fall back to the NumPy reference
    _flight_core = None


def flight_params(kd: float, floor: bool = True):
    return _flight_core.FlightParams(bal.GRAVITY_M_S2, kd, bal.BALL_RADIUS_M, bal.NOMINAL_RESTITUTION,
                                     bal.NOMINAL_TANGENTIAL_RETENTION, floor)


def propagate_reference(states: np.ndarray, dt: float, kd: float, residual=None) -> np.ndarray:
    """NumPy implementation. The compiled core is tested against this one."""
    steps = max(1, int(np.ceil(dt / MAX_SUBSTEP_S - 1e-9)))
    h = dt / steps
    p, v = states[:, :3], states[:, 3:]
    for _ in range(steps):
        p, v = bal.step(p, v, h, kd, residual)
    return np.hstack([p, v])


def propagate(states: np.ndarray, dt: float, kd: float, residual=None) -> np.ndarray:
    """Advance a batch of states (N, 6) by dt through the flight model.

    A learned residual is a Python callable, so that case stays in NumPy.
    """
    if _flight_core is None or residual is not None:
        return propagate_reference(states, dt, kd, residual)
    return _flight_core.propagate(states, dt, flight_params(kd), MAX_SUBSTEP_S)


def rollout(states: np.ndarray, dt: float, steps: int, kd: float, residual=None) -> np.ndarray:
    """States after dt, 2 dt, ..., steps dt. Shape (steps, N, 6)."""
    if _flight_core is None or residual is not None:
        out = np.empty((steps,) + states.shape)
        for k in range(steps):
            states = propagate_reference(states, dt, kd, residual)
            out[k] = states
        return out
    return _flight_core.rollout(states, dt, steps, flight_params(kd), MAX_SUBSTEP_S)


@dataclass
class Measurement:
    """Pixel observations of one stereo frame. A camera that saw nothing has pixel None."""
    pixels: tuple
    poses: tuple[CameraPose, CameraPose]
    # Optional: every detection in each image (a list of pixels per camera), for starting a track
    # when the strongest detection may not be the ball. pixels holds the strongest of each.
    candidates: tuple | None = None

    def stacked(self) -> np.ndarray:
        return np.concatenate([p for p in self.pixels if p is not None])

    def cameras(self) -> list[int]:
        return [i for i, p in enumerate(self.pixels) if p is not None]


class Ukf:
    def __init__(self, intr: Intrinsics, accel_noise_density: float, pixel_sigma: float,
                 kd: float = bal.NOMINAL_KD, residual=None):
        self.intr = intr
        self.q = accel_noise_density
        self.pixel_var = pixel_sigma**2
        self.kd = kd
        self.residual = residual
        self.x = np.zeros(STATE_DIM)
        self.P = np.eye(STATE_DIM)

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        pts, w = sigma_points(self.x, self.P)
        pts = propagate(pts, dt, self.kd, self.residual)
        self.x = w @ pts
        dev = pts - self.x
        self.P = (dev.T * w) @ dev + white_acceleration_noise(self.q, dt)

    def _measure(self, states: np.ndarray, meas: Measurement) -> np.ndarray:
        return np.hstack([project(self.intr, meas.poses[c], states[:, :3])[0] for c in meas.cameras()])

    def innovation(self, meas: Measurement):
        """Innovation, its covariance, and the pieces needed to finish the update."""
        pts, w = sigma_points(self.x, self.P)
        zs = self._measure(pts, meas)
        z_hat = w @ zs
        dz = zs - z_hat
        s = (dz.T * w) @ dz + self.pixel_var * np.eye(zs.shape[1])
        pxz = ((pts - self.x).T * w) @ dz
        return meas.stacked() - z_hat, s, pxz

    def update(self, meas: Measurement) -> tuple[float, float]:
        """Apply the measurement. Returns (NIS, Gaussian likelihood of the innovation)."""
        nu, s, pxz = self.innovation(meas)
        s_inv = np.linalg.inv(s)
        k = pxz @ s_inv
        self.x = self.x + k @ nu
        self.P = symmetric_positive(self.P - k @ s @ k.T)
        nis = float(nu @ s_inv @ nu)
        likelihood = float(np.exp(-0.5 * nis) / np.sqrt(np.linalg.det(2 * np.pi * s)))
        return nis, likelihood
