"""Will this throw be caught? A logistic model on what the planner knows when it commits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Commit-time features from CatchRunner._features, plus squares of the catch point's offset from
# the shoulder so that a linear model can represent "too far in any direction".
BASE_FEATURES = ("time_to_go", "slack", "reach", "rel_x", "rel_y", "rel_z", "sigma", "speed", "descent", "travel")
SQUARED_FEATURES = ("rel_x", "rel_y", "rel_z")
FEATURE_NAMES = BASE_FEATURES + tuple(f"{name}^2" for name in SQUARED_FEATURES)


def feature_vector(features: dict) -> np.ndarray:
    base = [features[name] for name in BASE_FEATURES]
    return np.array(base + [features[name] ** 2 for name in SQUARED_FEATURES])


@dataclass
class CatchGate:
    """Standardised logistic regression. Callable on the runtime's commit features."""
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5

    def probability(self, features: dict) -> float:
        z = (feature_vector(features) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.weights + self.bias))))

    def __call__(self, features: dict) -> bool:
        return self.probability(features) >= self.threshold

    @staticmethod
    def fit(feature_rows: list[dict], caught: np.ndarray, regularisation: float = 1.0) -> "CatchGate":
        from sklearn.linear_model import LogisticRegression
        x = np.array([feature_vector(f) for f in feature_rows])
        mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-9
        model = LogisticRegression(C=regularisation, max_iter=2000).fit((x - mean) / scale, caught)
        return CatchGate(mean, scale, model.coef_[0], float(model.intercept_[0]))
