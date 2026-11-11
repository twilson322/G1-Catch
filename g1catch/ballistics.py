"""Ball flight model shared by the filter, the predictor and the simulator's aerodynamics.

    v' = g - kd |v| v + km (omega x v)

kd = rho Cd A / (2 m) has units 1/m. The filter uses a nominal kd and no spin. The simulator
applies the true drag and Magnus force, which differ per throw.
"""
from __future__ import annotations

import numpy as np

GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.2
BALL_RADIUS_M = 0.033
BALL_MASS_KG = 0.057
NOMINAL_DRAG_COEFFICIENT = 0.47     # smooth sphere, subcritical Reynolds number
# Floor bounce used by the filter's process model. Measured on the simulator (tests/test_ballistics.py). Retention varies 0.84 to 0.89 with spin.
NOMINAL_RESTITUTION = 0.73
NOMINAL_TANGENTIAL_RETENTION = 0.86  # fraction of horizontal speed kept through a bounce

_G = np.array([0.0, 0.0, -GRAVITY_M_S2])


def drag_per_metre(cd: float, radius: float = BALL_RADIUS_M, mass: float = BALL_MASS_KG) -> float:
    """kd in v' = -kd |v| v."""
    return AIR_DENSITY_KG_M3 * cd * np.pi * radius**2 / (2.0 * mass)


def magnus_per_radian(radius: float = BALL_RADIUS_M, mass: float = BALL_MASS_KG) -> float:
    """km in v' = km (omega x v), from a lift coefficient equal to the spin ratio r w / |v|."""
    return 0.5 * AIR_DENSITY_KG_M3 * np.pi * radius**3 / mass


NOMINAL_KD = drag_per_metre(NOMINAL_DRAG_COEFFICIENT)


def acceleration(v: np.ndarray, kd: float = NOMINAL_KD, residual=None) -> np.ndarray:
    """Model acceleration for velocities of shape (..., 3). `residual(v)` adds a learned term."""
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a = _G - kd * speed * v
    if residual is not None:
        a = a + residual(v)
    return a


def step(p: np.ndarray, v: np.ndarray, dt: float, kd: float = NOMINAL_KD, residual=None,
         floor: bool = True):
    """One RK4 step of batched states (..., 3), with an instantaneous floor bounce.

    The bounce is applied after the step to any state that ended below the floor while moving
    down. That places the impact up to one step late, which the bounce-mode process noise covers.
    """
    def f(vel):
        return acceleration(vel, kd, residual)
    k1v = f(v)
    k2v = f(v + 0.5 * dt * k1v)
    k3v = f(v + 0.5 * dt * k2v)
    k4v = f(v + dt * k3v)
    p_new = p + dt * (v + dt / 6.0 * (k1v + k2v + k3v))
    v_new = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    if floor:
        hit = (p_new[..., 2] < BALL_RADIUS_M) & (v_new[..., 2] < 0.0)
        if np.any(hit):
            p_new = p_new.copy()
            v_new = v_new.copy()
            p_new[hit, 2] = 2.0 * BALL_RADIUS_M - p_new[hit, 2]
            v_new[hit, 2] *= -NOMINAL_RESTITUTION
            v_new[hit, :2] *= NOMINAL_TANGENTIAL_RETENTION
    return p_new, v_new


def rollout(p0: np.ndarray, v0: np.ndarray, dt: float, steps: int, kd: float = NOMINAL_KD,
            residual=None, floor: bool = True):
    """Positions and velocities at t = dt, 2 dt, ..., steps*dt. Output shapes (steps, ..., 3)."""
    p = np.asarray(p0, dtype=float)
    v = np.asarray(v0, dtype=float)
    ps = np.empty((steps,) + p.shape)
    vs = np.empty((steps,) + v.shape)
    for k in range(steps):
        p, v = step(p, v, dt, kd, residual, floor)
        ps[k], vs[k] = p, v
    return ps, vs
