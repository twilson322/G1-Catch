"""Joint sensing and state observers for the torque loop.

A real joint reports a quantized angle and no velocity. After computed torque each joint is the
double integrator q'' = u + d, where u is the commanded acceleration and d lumps everything the
model got wrong (link masses, friction, an unknown payload). The observers here estimate
velocity, and optionally d, from the quantized angle and the acceleration that was applied.

    exact       true angle and velocity from the simulator (the idealised baseline)
    difference  quantized angle, velocity by backward difference and a first-order low-pass
    kalman      steady-state Kalman filter on [q, q']: with the LQR gains this is an LQG loop
    eso         steady-state Kalman filter on [q, q', d]: an extended-state (disturbance) observer
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

OBSERVERS = ("exact", "difference", "kalman", "eso")


@dataclass
class JointSensingConfig:
    observer: str = "kalman"
    encoder_bits: int = 15                    # counts per revolution = 2^bits (0.19 mrad at 15)
    difference_cutoff_hz: float = 80.0
    # Kalman design. Process noise is acceleration noise (rad/s^2) and, for the ESO, the rate at
    # which the disturbance may wander (rad/s^3). Larger values trust the encoder more.
    accel_noise: float = 30.0
    disturbance_rate_noise: float = 1000.0    # 28 ms rise, 0.7 rad/s^2 of estimate noise at 15 bits


def quantize(q: np.ndarray, bits: int) -> np.ndarray:
    step = 2.0 * np.pi / 2**bits
    return np.round(q / step) * step


def observer_model(period: float, with_disturbance: bool):
    """Discrete model of one joint: x+ = A x + B u, y = C x. State [q, q'] or [q, q', d]."""
    if with_disturbance:
        a = np.array([[1.0, period, 0.5 * period**2], [0.0, 1.0, period], [0.0, 0.0, 1.0]])
        b = np.array([0.5 * period**2, period, 0.0])
        c = np.array([[1.0, 0.0, 0.0]])
    else:
        a = np.array([[1.0, period], [0.0, 1.0]])
        b = np.array([0.5 * period**2, period])
        c = np.array([[1.0, 0.0]])
    return a, b, c


def kalman_gain(period: float, cfg: JointSensingConfig, with_disturbance: bool) -> np.ndarray:
    """Steady-state measurement-update gain L, from the filter Riccati equation.

    Measurement noise is the quantization error, uniform over one count: variance step^2 / 12.
    """
    a, b, c = observer_model(period, with_disturbance)
    step = 2.0 * np.pi / 2**cfg.encoder_bits
    r = np.array([[step**2 / 12.0]])
    g = b[:2] if not with_disturbance else b            # acceleration noise enters like the input
    q = np.outer(g, g) * cfg.accel_noise**2
    if with_disturbance:
        q = q + np.diag([0.0, 0.0, (cfg.disturbance_rate_noise * period) ** 2])
    else:
        q = q + 1e-18 * np.eye(2)
    p = solve_discrete_are(a.T, c.T, q, r)
    return (p @ c.T / (c @ p @ c.T + r)).ravel()


class JointObserver:
    """The same scalar observer run on every joint. Call `correct` with a measurement, then
    `predict` with the acceleration that was actually applied."""

    def __init__(self, n_joints: int, period: float, cfg: JointSensingConfig | None = None):
        self.cfg = cfg or JointSensingConfig()
        if self.cfg.observer not in OBSERVERS:
            raise ValueError(f"unknown observer {self.cfg.observer!r}, expected one of {OBSERVERS}")
        self.n = n_joints
        self.period = period
        self.q = np.zeros(n_joints)
        self.dq = np.zeros(n_joints)
        self.disturbance = np.zeros(n_joints)
        kind = self.cfg.observer
        self._model = None
        if kind in ("kalman", "eso"):
            with_d = kind == "eso"
            self._model = observer_model(period, with_d)
            self.gain = kalman_gain(period, self.cfg, with_d)
        self._alpha = 1.0 - np.exp(-2.0 * np.pi * self.cfg.difference_cutoff_hz * period)
        self._previous = np.zeros(n_joints)
        self._started = False

    def reset(self, q: np.ndarray) -> None:
        self.q = q.copy()
        self.dq = np.zeros(self.n)
        self.disturbance = np.zeros(self.n)
        self._previous = self._measure(q)
        self._started = True

    def _measure(self, q_true: np.ndarray) -> np.ndarray:
        return quantize(q_true, self.cfg.encoder_bits)

    def correct(self, q_true: np.ndarray, dq_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Take this tick's encoder reading. Returns the angle and velocity estimates."""
        kind = self.cfg.observer
        if not self._started:
            self.reset(q_true)          # a first reading with no reset: start from it, at rest
        if kind == "exact":
            self.q, self.dq = q_true.copy(), dq_true.copy()
            return self.q, self.dq
        y = self._measure(q_true)
        if kind == "difference":
            self.dq = self.dq + self._alpha * ((y - self._previous) / self.period - self.dq)
            self._previous = y
            self.q = y
            return self.q, self.dq
        innovation = y - self.q
        self.q = self.q + self.gain[0] * innovation
        self.dq = self.dq + self.gain[1] * innovation
        if kind == "eso":
            self.disturbance = self.disturbance + self.gain[2] * innovation
        return self.q, self.dq

    def predict(self, applied_accel: np.ndarray) -> None:
        """Advance the model one period with the acceleration the controller actually applied."""
        if self._model is None:
            return
        t = self.period
        total = applied_accel + self.disturbance
        self.q = self.q + t * self.dq + 0.5 * t**2 * total
        self.dq = self.dq + t * total
