"""Shrinking-horizon MPC for the reach, as a QP over joint accelerations.

Computed torque makes every joint a double integrator, so the dynamics are linear and the joints
couple only through the shared horizon. The terminal state is the catch configuration at the
catch time, held softly so the problem stays feasible when the target jumps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp

from .sim import ARM_DOF


@dataclass
class MpcConfig:
    max_steps: int = 14
    min_step_s: float = 0.0165          # one camera frame
    effort_weight: float = 1.0
    jerk_weight: float = 2.0            # on the change in acceleration between steps
    terminal_position_weight: float = 4.0e7
    terminal_velocity_weight: float = 2.0e4
    # Limits. Acceleration bounds stand in for torque limits, see scripts/ilqr_check.py.
    accel_limit: np.ndarray = field(default_factory=lambda: np.array([25.0, 25.0, 25.0, 70.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0]))
    velocity_limit: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0, 11.0, 11.0, 11.0, 11.0, 14.0, 14.0, 14.0]))
    joint_margin_rad: float = 0.05


@dataclass
class MpcSolution:
    accels: np.ndarray      # (steps, ARM_DOF)
    step: float
    solved: bool
    terminal_position_error: np.ndarray
    solve_time_s: float


def _horizon(time_to_go: float, cfg: MpcConfig) -> tuple[int, float]:
    steps = int(np.clip(np.ceil(time_to_go / cfg.min_step_s), 1, cfg.max_steps))
    return steps, time_to_go / steps


def prediction_matrices(steps: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Sq, Sv with q_k = q0 + k h v0 + Sq[k-1] @ u and v_k = v0 + Sv[k-1] @ u for k = 1..steps."""
    sv = h * np.tril(np.ones((steps, steps)))
    k = np.arange(1, steps + 1)[:, None]
    i = np.arange(steps)[None, :]
    sq = h**2 * np.where(i < k, k - i - 0.5, 0.0)
    return sq, sv


class ReachMpc:
    def __init__(self, joint_lo: np.ndarray, joint_hi: np.ndarray, cfg: MpcConfig | None = None):
        self.cfg = cfg or MpcConfig()
        self.q_lo = joint_lo + self.cfg.joint_margin_rad
        self.q_hi = joint_hi - self.cfg.joint_margin_rad
        self._solvers: dict[int, osqp.OSQP] = {}

    def solve(self, q0, dq0, q_goal, dq_goal, time_to_go: float, previous_accel=None) -> MpcSolution:
        cfg = self.cfg
        n, h = _horizon(time_to_go, cfg)
        sq, sv = prediction_matrices(n, h)
        prev = np.zeros(ARM_DOF) if previous_accel is None else previous_accel

        # One joint's cost: effort, jerk, terminal position and velocity.
        diff = np.eye(n) - np.eye(n, k=-1)
        p1 = 2.0 * (cfg.effort_weight * np.eye(n) + cfg.jerk_weight * diff.T @ diff
                    + cfg.terminal_position_weight * np.outer(sq[-1], sq[-1])
                    + cfg.terminal_velocity_weight * np.outer(sv[-1], sv[-1]))
        a1 = np.vstack([np.eye(n), sv, sq])
        p = sp.kron(sp.eye(ARM_DOF), sp.csc_matrix(np.triu(p1)), format="csc")
        a = sp.kron(sp.eye(ARM_DOF), sp.csc_matrix(a1), format="csc")

        k = np.arange(1, n + 1)
        lin = np.empty(ARM_DOF * n)
        lo = np.empty(ARM_DOF * 3 * n)
        hi = np.empty(ARM_DOF * 3 * n)
        first = np.zeros(n)
        first[0] = 1.0
        for j in range(ARM_DOF):
            free_q = q0[j] + k * h * dq0[j]          # position with zero acceleration
            lin[j * n:(j + 1) * n] = 2.0 * (
                cfg.terminal_position_weight * (free_q[-1] - q_goal[j]) * sq[-1]
                + cfg.terminal_velocity_weight * (dq0[j] - dq_goal[j]) * sv[-1]
                - cfg.jerk_weight * prev[j] * first)
            row = j * 3 * n
            lo[row:row + n] = -cfg.accel_limit[j]
            hi[row:row + n] = cfg.accel_limit[j]
            lo[row + n:row + 2 * n] = -cfg.velocity_limit[j] - dq0[j]
            hi[row + n:row + 2 * n] = cfg.velocity_limit[j] - dq0[j]
            # A start outside the margin must not make the problem infeasible.
            lo[row + 2 * n:row + 3 * n] = np.minimum(self.q_lo[j], q0[j]) - free_q
            hi[row + 2 * n:row + 3 * n] = np.maximum(self.q_hi[j], q0[j]) - free_q

        # One solver per horizon length. The sparsity pattern depends only on the number of steps,
        # so later frames with the same length update the numbers in place and start from the
        # previous solution, which skips the setup and most of the iterations.
        solver = self._solvers.get(n)
        if solver is None:
            solver = osqp.OSQP()
            solver.setup(p, lin, a, lo, hi, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, polish=False,
                         warm_starting=True)
            self._solvers[n] = solver
        else:
            solver.update(q=lin, l=lo, u=hi, Px=p.data, Ax=a.data)
        res = solver.solve()
        solved = res.info.status in ("solved", "solved inaccurate") and res.x is not None and np.all(np.isfinite(res.x))
        if solved:
            u = res.x.reshape(ARM_DOF, n).T
            u = np.clip(u, -cfg.accel_limit, cfg.accel_limit)
        else:
            u = np.zeros((n, ARM_DOF))
        q_end = q0 + n * h * dq0 + u.T @ sq[-1]
        return MpcSolution(u, h, solved, q_end - q_goal, res.info.run_time if solved else 0.0)
