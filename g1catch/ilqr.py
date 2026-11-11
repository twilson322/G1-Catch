"""Iterative LQR with box limits on the controls.

Minimises  sum_k 1/2 u_k' R u_k  +  1/2 (x_N - x_goal)' Qf (x_N - x_goal)
subject to x_{k+1} = f(x_k, u_k) and lo <= u_k <= hi. The dynamics and their derivatives are
supplied by the caller, so the same solver runs on a linear test system and on MuJoCo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class IlqrOptions:
    max_iterations: int = 60
    cost_tolerance: float = 1e-6        # relative cost decrease that counts as converged
    initial_regularisation: float = 1e-6
    regularisation_growth: float = 10.0
    max_regularisation: float = 1e6
    line_search_steps: tuple = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01)


@dataclass
class IlqrResult:
    states: np.ndarray                  # (N+1, nx)
    controls: np.ndarray                # (N, nu)
    gains: np.ndarray                   # (N, nu, nx) feedback about the trajectory
    cost: float
    iterations: int
    converged: bool
    cost_history: list = field(default_factory=list)


def _total_cost(xs, us, r_diag, qf, x_goal) -> float:
    dx = xs[-1] - x_goal
    return 0.5 * float(np.sum(us**2 * r_diag) + dx @ qf @ dx)


def _clamped_step(quu, qu, qux, u, lo, hi):
    """Feedforward k and gain K minimising the local model subject to lo <= u + k <= hi.

    One projection pass: solve unconstrained, clamp the controls that leave the box, then solve
    again for the rest with the clamped ones held at their bounds.
    """
    k = -np.linalg.solve(quu, qu)
    clamped = ((u + k > hi) & (k > 0)) | ((u + k < lo) & (k < 0))
    if not np.any(clamped):
        return k, -np.linalg.solve(quu, qux)
    free = ~clamped
    k = np.where(clamped, np.clip(u + k, lo, hi) - u, 0.0)
    gain = np.zeros_like(qux)
    if np.any(free):
        quu_ff = quu[np.ix_(free, free)]
        k[free] = -np.linalg.solve(quu_ff, qu[free] + quu[np.ix_(free, clamped)] @ k[clamped])
        gain[free] = -np.linalg.solve(quu_ff, qux[free])
    return k, gain


def solve(step: Callable, linearise: Callable, x0: np.ndarray, controls: np.ndarray, x_goal: np.ndarray,
          r_diag: np.ndarray, qf: np.ndarray, lo: np.ndarray, hi: np.ndarray,
          opt: IlqrOptions | None = None) -> IlqrResult:
    """step(x, u) -> x_next. linearise(x, u) -> (A, B). controls: initial guess (N, nu)."""
    opt = opt or IlqrOptions()
    n, nu = controls.shape
    nx = x0.size

    def rollout(us):
        xs = np.empty((n + 1, nx))
        xs[0] = x0
        for k in range(n):
            xs[k + 1] = step(xs[k], us[k])
        return xs

    us = np.clip(controls, lo, hi)
    xs = rollout(us)
    cost = _total_cost(xs, us, r_diag, qf, x_goal)
    history = [cost]
    reg = opt.initial_regularisation
    gains = np.zeros((n, nu, nx))
    converged = False
    iteration = 0
    for iteration in range(1, opt.max_iterations + 1):
        # Backward pass.
        derivs = [linearise(xs[k], us[k]) for k in range(n)]
        while True:
            vx = qf @ (xs[-1] - x_goal)
            vxx = qf.copy()
            ks = np.zeros((n, nu))
            ok = True
            for k in range(n - 1, -1, -1):
                a, b = derivs[k]
                qx = a.T @ vx
                qu = r_diag * us[k] + b.T @ vx
                qxx = a.T @ vxx @ a
                qux = b.T @ vxx @ a
                quu = np.diag(r_diag) + b.T @ vxx @ b + reg * np.eye(nu)
                if np.any(np.linalg.eigvalsh(quu) <= 0):
                    ok = False
                    break
                ks[k], gains[k] = _clamped_step(quu, qu, qux, us[k], lo, hi)
                vx = qx + gains[k].T @ quu @ ks[k] + gains[k].T @ qu + qux.T @ ks[k]
                vxx = qxx + gains[k].T @ quu @ gains[k] + gains[k].T @ qux + qux.T @ gains[k]
                vxx = 0.5 * (vxx + vxx.T)
            if ok:
                break
            reg *= opt.regularisation_growth
            if reg > opt.max_regularisation:
                return IlqrResult(xs, us, gains, cost, iteration, False, history)

        # Forward pass with a line search on the feedforward term.
        improved = False
        for alpha in opt.line_search_steps:
            new_xs = np.empty_like(xs)
            new_us = np.empty_like(us)
            new_xs[0] = x0
            for k in range(n):
                new_us[k] = np.clip(us[k] + alpha * ks[k] + gains[k] @ (new_xs[k] - xs[k]), lo, hi)
                new_xs[k + 1] = step(new_xs[k], new_us[k])
            new_cost = _total_cost(new_xs, new_us, r_diag, qf, x_goal)
            if np.isfinite(new_cost) and new_cost < cost:
                improved = True
                break
        if not improved:
            reg *= opt.regularisation_growth
            if reg > opt.max_regularisation:
                break
            continue
        decrease = (cost - new_cost) / max(cost, 1e-12)
        xs, us, cost = new_xs, new_us, new_cost
        history.append(cost)
        reg = max(reg / opt.regularisation_growth, opt.initial_regularisation)
        if decrease < opt.cost_tolerance:
            converged = True
            break
    return IlqrResult(xs, us, gains, cost, iteration, converged, history)
