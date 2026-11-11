"""A neural-network stand-in for the reach MPC, trained by imitation (DAgger).

The MPC's cost and constraints are separable by joint, so the policy is one small network applied
to each joint independently: it sees that joint's error, velocity, goal velocity, time to go,
previous command and distance to its limits, and returns the next acceleration. Rolling the
network forward on the double-integrator model gives a full plan with the same interface as the
MPC, so the two are interchangeable in the runtime.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .mpc import MpcConfig, MpcSolution, ReachMpc, _horizon
from .safety import filter_acceleration
from .sim import ARM_DOF

FEATURES = 8
TIME_SCALE_S = 0.4


def joint_features(q, dq, q_goal, dq_goal, time_to_go, previous_accel, q_lo, q_hi, cfg: MpcConfig) -> np.ndarray:
    """(ARM_DOF, FEATURES) inputs, scaled to order one by each joint's own limits."""
    a_max, v_max = cfg.accel_limit, cfg.velocity_limit
    reach_scale = v_max * TIME_SCALE_S
    t = np.full(ARM_DOF, time_to_go / TIME_SCALE_S)
    return np.column_stack([
        (q_goal - q) / reach_scale, dq / v_max, dq_goal / v_max, t, 1.0 / (t + 0.05),
        previous_accel / a_max, (q - q_lo) / reach_scale, (q_hi - q) / reach_scale]).astype(np.float32)


class JointPolicyNet(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Acceleration as a fraction of the joint's limit, in [-1, 1]."""
        return self.net(x).squeeze(-1)


class PolicyPlanner:
    """Drop-in replacement for ReachMpc.solve that rolls a JointPolicyNet forward to the catch time."""

    def __init__(self, net: JointPolicyNet, joint_lo, joint_hi, cfg: MpcConfig | None = None,
                 safety_filter: bool = True):
        self.net = net.eval()
        self.safety_filter = safety_filter
        self.filtered_steps = 0
        self.total_steps = 0
        self.cfg = cfg or MpcConfig()
        self.q_lo = joint_lo + self.cfg.joint_margin_rad
        self.q_hi = joint_hi - self.cfg.joint_margin_rad

    @torch.no_grad()
    def first_accel(self, q, dq, q_goal, dq_goal, time_to_go, previous_accel) -> np.ndarray:
        x = joint_features(q, dq, q_goal, dq_goal, time_to_go, previous_accel, self.q_lo, self.q_hi, self.cfg)
        return self.net(torch.from_numpy(x)).numpy() * self.cfg.accel_limit

    def solve(self, q0, dq0, q_goal, dq_goal, time_to_go: float, previous_accel=None) -> MpcSolution:
        t0 = time.perf_counter()
        n, h = _horizon(time_to_go, self.cfg)
        prev = np.zeros(ARM_DOF) if previous_accel is None else previous_accel
        q, dq = q0.copy(), dq0.copy()
        accels = np.empty((n, ARM_DOF))
        for k in range(n):
            u = self.first_accel(q, dq, q_goal, dq_goal, time_to_go - k * h, prev)
            if self.safety_filter:
                # The network has no notion of a limit. The barrier filter gives it one.
                u, changed = filter_acceleration(q, dq, u, h, self.q_lo, self.q_hi, self.cfg.accel_limit,
                                                 self.cfg.velocity_limit)
                self.filtered_steps += int(np.any(changed))
            else:
                u = np.clip(u, (-self.cfg.velocity_limit - dq) / h, (self.cfg.velocity_limit - dq) / h)
            self.total_steps += 1
            accels[k] = u
            q = q + h * dq + 0.5 * h**2 * u
            dq = dq + h * u
            prev = u
        return MpcSolution(accels, h, True, q - q_goal, time.perf_counter() - t0)


@dataclass
class Dataset:
    features: list
    targets: list

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self.features.append(x)
        self.targets.append(y)

    def arrays(self):
        return np.concatenate(self.features), np.concatenate(self.targets)


class DaggerPlanner:
    """Runs the learner (or the expert) and labels every visited state with the expert's action."""

    def __init__(self, expert: ReachMpc, learner: PolicyPlanner | None, dataset: Dataset):
        self.expert, self.learner, self.dataset = expert, learner, dataset
        self.cfg = expert.cfg

    def solve(self, q0, dq0, q_goal, dq_goal, time_to_go: float, previous_accel=None) -> MpcSolution:
        prev = np.zeros(ARM_DOF) if previous_accel is None else previous_accel
        expert_sol = self.expert.solve(q0, dq0, q_goal, dq_goal, time_to_go, prev)
        if expert_sol.solved:
            x = joint_features(q0, dq0, q_goal, dq_goal, time_to_go, prev, self.expert.q_lo, self.expert.q_hi, self.cfg)
            self.dataset.add(x, (expert_sol.accels[0] / self.cfg.accel_limit).astype(np.float32))
        if self.learner is None:
            return expert_sol
        return self.learner.solve(q0, dq0, q_goal, dq_goal, time_to_go, prev)


def fit(net: JointPolicyNet, dataset: Dataset, epochs: int = 30, seed: int = 0) -> float:
    """Supervised regression on everything collected so far. Returns the final training loss."""
    torch.manual_seed(seed)
    x, y = (torch.from_numpy(a) for a in dataset.arrays())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    loss_value = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for i in range(0, len(x), 512):
            idx = order[i:i + 512]
            loss = torch.mean((net(x[idx]) - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_value = float(loss)
    net.eval()
    return loss_value
