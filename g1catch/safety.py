"""A safety filter for joint limits, built on a braking-distance control barrier function.

For a joint that can decelerate at a_brake, the state (q, q') is safe with respect to the upper
limit if it can still stop in time:

    h(q, q') = q_hi - q - max(q', 0)^2 / (2 a_brake) >= 0,

and likewise for the lower limit. The filter takes any commanded acceleration and returns the
closest one that keeps h >= 0 after the next step. It never changes a command that is already
safe, so it can sit behind a planner with no guarantees of its own (the distilled network).
"""
from __future__ import annotations

import numpy as np

def _fastest_safe_velocity(gap: np.ndarray, dq: np.ndarray, step: float, a_brake: np.ndarray) -> np.ndarray:
    """Largest velocity toward a limit, one step from now, from which the joint can still stop.

    gap is the distance to the limit now, dq the velocity toward it. With next velocity w the
    joint moves step (dq + w) / 2 during the step and then needs w^2 / (2 a_brake) to stop:

        step (dq + w) / 2 + w^2 / (2 a_brake) <= gap     for w >= 0.

    That is a quadratic in w. If it has no non-negative solution the joint must already be moving
    away at the end of the step, and the constraint is the linear one without the braking term.
    """
    c = 0.5 * step * dq - gap
    disc = 0.25 * step**2 - 2.0 * c / a_brake
    w_quadratic = a_brake * (-0.5 * step + np.sqrt(np.maximum(disc, 0.0)))
    w_linear = -2.0 * c / step
    return np.where(c < 0.0, w_quadratic, w_linear)


def filter_acceleration(q, dq, u, step, q_lo, q_hi, accel_limit, velocity_limit, brake_fraction: float = 0.9):
    """The acceleration closest to `u` that respects acceleration, velocity and position limits.

    All arguments are per-joint arrays except `step` (s) and brake_fraction, the share of the
    acceleration limit the barrier assumes is available for braking.
    Returns (filtered acceleration, boolean mask of the joints that were changed).
    """
    a_brake = brake_fraction * accel_limit
    lo = np.maximum(-accel_limit, (-velocity_limit - dq) / step)
    hi = np.minimum(accel_limit, (velocity_limit - dq) / step)
    hi = np.maximum(hi, lo)

    u_max = np.clip((_fastest_safe_velocity(q_hi - q, dq, step, a_brake) - dq) / step, lo, hi)
    u_min = np.clip((-_fastest_safe_velocity(q - q_lo, -dq, step, a_brake) - dq) / step, lo, hi)
    u_min = np.minimum(u_min, u_max)        # both limits active at once: the upper one wins
    filtered = np.clip(u, u_min, u_max)
    return filtered, np.abs(filtered - u) > 1e-9
