"""Learned corrections to the ball's acceleration model: a Gaussian process and boosted trees.

Both map velocity (3) to residual acceleration (3). The GP is fitted with scikit-learn and then
evaluated by a few lines of NumPy, because the filter calls it thousands of times per frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


@dataclass
class GpResidual:
    """Posterior mean of a GP with an anisotropic RBF kernel: a(v) = K(v, X) alpha."""
    inputs: np.ndarray           # (N, 3) training velocities
    alpha: np.ndarray            # (N, 3)
    length_scale: np.ndarray     # (3,)
    signal_variance: float
    noise_variance: float
    chol_inv: np.ndarray | None = None   # L^-1 for predictive variance, optional

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self._kernel(v) @ self.alpha

    def _kernel(self, v: np.ndarray) -> np.ndarray:
        diff = (np.atleast_2d(v)[:, None, :] - self.inputs[None, :, :]) / self.length_scale
        return self.signal_variance * np.exp(-0.5 * np.sum(diff**2, axis=2))

    def std(self, v: np.ndarray) -> np.ndarray:
        """Predictive standard deviation of the latent function, per query point."""
        k = self._kernel(v)
        proj = k @ self.chol_inv.T
        return np.sqrt(np.maximum(self.signal_variance - np.sum(proj**2, axis=1), 0.0))

    @staticmethod
    def fit(velocities: np.ndarray, residuals: np.ndarray, max_points: int = 400, seed: int = 0) -> "GpResidual":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(velocities), size=min(max_points, len(velocities)), replace=False)
        x, y = velocities[idx], residuals[idx]
        kernel = ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[3.0, 3.0, 3.0], length_scale_bounds=(0.5, 50.0)) \
            + WhiteKernel(0.5, (1e-4, 1e2))
        gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=2, random_state=seed).fit(x, y)
        fitted = gp.kernel_
        signal = float(fitted.k1.k1.constant_value)
        scale = np.asarray(fitted.k1.k2.length_scale, dtype=float)
        noise = float(fitted.k2.noise_level)
        return GpResidual(x, gp.alpha_, scale, signal, noise, np.linalg.inv(gp.L_))


class TreeResidual:
    """Gradient-boosted regression trees, one model per acceleration component."""

    def __init__(self, models):
        self.models = models

    def __call__(self, v: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(v)
        return np.column_stack([m.predict(v) for m in self.models])

    @staticmethod
    def fit(velocities: np.ndarray, residuals: np.ndarray, seed: int = 0) -> "TreeResidual":
        models = [GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                            random_state=seed).fit(velocities, residuals[:, i]) for i in range(3)]
        return TreeResidual(models)
