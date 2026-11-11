"""Online payload estimation by recursive least squares, and its use in the arm model.

The wrist force sensor reads f = (m_cup + m_ball)(a - g) in the world frame, with a the cup's
acceleration and g the gravity vector. The cup's mass is known, so

    y = f - m_cup (a - g) = (a - g) m_ball

is linear in the one unknown. The estimate is written into the arm model, where it corrects the
mass matrix and the gravity torque (an indirect, self-tuning adaptive controller).
"""
from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])
CUP_MASS_KG = 0.09


class ScalarRls:
    """Recursive least squares for y = phi * theta with vector y and phi, scalar theta."""

    def __init__(self, initial: float = 0.0, initial_variance: float = 1.0, forgetting: float = 0.995,
                 max_variance: float = 10.0):
        self.theta = initial
        self.variance = initial_variance
        self.forgetting = forgetting
        self.max_variance = max_variance

    def update(self, phi: np.ndarray, y: np.ndarray) -> float:
        excitation = float(phi @ phi)
        gain = self.variance * phi / (self.forgetting + self.variance * excitation)
        self.theta += float(gain @ (y - phi * self.theta))
        self.variance = min((self.variance - self.variance * float(gain @ phi)) / self.forgetting, self.max_variance)
        return self.theta


class PayloadEstimator:
    def __init__(self, forgetting: float = 0.995, min_mass: float = 0.0, max_mass: float = 2.0):
        self.rls = ScalarRls(forgetting=forgetting)
        self.min_mass, self.max_mass = min_mass, max_mass

    def update(self, force_world: np.ndarray, cup_acceleration: np.ndarray) -> float:
        phi = cup_acceleration - GRAVITY
        y = force_world - CUP_MASS_KG * phi
        return float(np.clip(self.rls.update(phi, y), self.min_mass, self.max_mass))
