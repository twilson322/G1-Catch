"""Choosing where and when to meet the ball, and the arm configuration that does it."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .control import ArmModel, axis_error
from .sim import ARM_DOF, READY_POSTURE


@dataclass
class InterceptConfig:
    shoulder: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.10, 1.085]))
    reach_min_m: float = 0.22
    reach_max_m: float = 0.52            # the cup centre reaches 0.55 m with the arm straight
    keep_margin_m: float = 0.025         # a committed point may drift this far outside before it is dropped
    min_forward_m: float = 0.18          # stay in front of the torso
    hand_accel_m_s2: float = 35.0        # for the minimum-time feasibility bound
    min_time_to_go_s: float = 0.02
    uncertainty_weight: float = 1.0      # metres of travel traded per metre of forecast sigma
    edge_weight: float = 1.0             # cost for catching near the workspace boundary
    infeasible_penalty: float = 10.0     # metres of travel per second short of the minimum time
    time_change_penalty_m_per_s: float = 3.0   # metres of travel per second the catch time is moved
    candidate_spacing_s: float = 0.02    # alternatives offered to the IK check are this far apart
    max_ik_error_m: float = 0.015        # a candidate the arm cannot reach this closely is dropped
    velocity_match: float = 0.45         # cup speed at the catch as a fraction of ball speed
    max_cup_speed_m_s: float = 2.0
    # A cup aimed straight against a rising ball opens downward and cannot keep it. The aim is
    # tilted up until its vertical component is at least this, and rising catch points cost
    # rising_penalty_s metres of travel per m/s of upward ball speed.
    min_axis_up: float = 0.35
    rising_penalty_s: float = 0.15


def catch_axis(ball_velocity: np.ndarray, min_up: float) -> np.ndarray:
    """Direction the cup should open: against the ball's velocity, but never tilted below `min_up`.

    min_up is the smallest allowed vertical component of the unit axis. For a falling ball the
    reversed velocity already points up and is returned unchanged.
    """
    axis = -ball_velocity / np.linalg.norm(ball_velocity)
    if axis[2] >= min_up:
        return axis
    horizontal = axis[:2]
    norm = np.linalg.norm(horizontal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    horizontal = horizontal / norm * np.sqrt(1.0 - min_up**2)
    return np.array([horizontal[0], horizontal[1], min_up])


@dataclass
class Intercept:
    time: float
    position: np.ndarray
    ball_velocity: np.ndarray
    position_sigma: float
    feasible: bool
    slack_s: float               # time available minus the minimum-time bound


def choose_intercept(times, positions, velocities, covariances, now: float, hand: np.ndarray,
                     cfg: InterceptConfig) -> Intercept | None:
    """The best point of the forecast to catch at. None if the ball never enters the workspace."""
    ranked = rank_intercepts(times, positions, velocities, covariances, now, hand, cfg, count=1)
    return ranked[0] if ranked else None


def rank_intercepts(times, positions, velocities, covariances, now: float, hand: np.ndarray,
                    cfg: InterceptConfig, count: int = 4, near_time: float | None = None) -> list[Intercept]:
    """Up to `count` candidate catch points, best first, at least cfg.candidate_spacing_s apart.

    The workspace test here is a sphere shell around the shoulder. It does not know about joint
    limits or the torso, so the caller checks each candidate with inverse kinematics and takes
    the first that works. near_time, when given, is a catch time already committed to: moving
    away from it costs cfg.time_change_penalty_m_per_s, so a re-choice stays close to the motion
    already under way.
    """
    rel = positions - cfg.shoulder
    dist = np.linalg.norm(rel, axis=1)
    time_to_go = times - now
    inside = (dist > cfg.reach_min_m) & (dist < cfg.reach_max_m) & (positions[:, 0] > cfg.min_forward_m) \
        & (time_to_go > cfg.min_time_to_go_s)
    if not np.any(inside):
        return []
    travel = np.linalg.norm(positions - hand, axis=1)
    # Rest-to-rest minimum time under an acceleration bound: accelerate half way, brake half way.
    t_min = 2.0 * np.sqrt(travel / cfg.hand_accel_m_s2)
    slack = time_to_go - t_min
    sigma = np.sqrt(np.trace(covariances, axis1=1, axis2=2))
    mid = 0.5 * (cfg.reach_min_m + cfg.reach_max_m)
    half = 0.5 * (cfg.reach_max_m - cfg.reach_min_m)

    def describe(i: int) -> Intercept:
        return Intercept(float(times[i]), positions[i].copy(), velocities[i].copy(), float(sigma[i]),
                         bool(slack[i] > 0.0), float(slack[i]))

    # Least hand travel, then forecast uncertainty, then distance from the workspace boundary.
    cost = travel + cfg.uncertainty_weight * sigma + cfg.edge_weight * ((dist - mid) / half) ** 2 * half
    cost = cost + cfg.infeasible_penalty * np.maximum(-slack, 0.0)
    cost = cost + cfg.rising_penalty_s * np.maximum(velocities[:, 2], 0.0)
    if near_time is not None:
        cost = cost + cfg.time_change_penalty_m_per_s * np.abs(times - near_time)
    cost[~inside] = np.inf
    picked: list[int] = []
    for i in np.argsort(cost):
        if not np.isfinite(cost[i]) or len(picked) == count:
            break
        if all(abs(times[i] - times[j]) >= cfg.candidate_spacing_s for j in picked):
            picked.append(int(i))
    return [describe(i) for i in picked]


def refresh_intercept(plan: Intercept, position, velocity, covariance, now: float,
                      cfg: InterceptConfig) -> Intercept | None:
    """The committed catch time with a new forecast of where the ball will be then.

    The time is kept fixed after commit. Re-choosing it every frame lets it creep along the
    ball's path, and the cup then meets the ball at an uncontrolled relative speed, or retreats
    in front of it indefinitely. Returns None if the ball is no longer forecast to be reachable
    at that time.
    """
    dist = np.linalg.norm(position - cfg.shoulder)
    slack = cfg.keep_margin_m       # hysteresis: a point on the boundary must not flip the plan
    if not (cfg.reach_min_m - slack < dist < cfg.reach_max_m + slack and position[0] > cfg.min_forward_m - slack):
        return None
    return Intercept(plan.time, position.copy(), velocity.copy(), float(np.sqrt(np.trace(covariance))),
                     plan.feasible, plan.time - now)


@dataclass
class IkConfig:
    iterations: int = 12
    polish_iterations: int = 20          # task-only iterations after a solve that used the posture pull
    position_damping: float = 0.02
    tilt_damping: float = 0.15
    posture_gain: float = 0.15
    max_step_rad: float = 0.35
    converged_step_rad: float = 1e-5     # stop iterating when the largest joint step is below this
    joint_margin_rad: float = 0.06
    # The waist is heavy and carries the cameras. Penalise its use.
    joint_weights: np.ndarray = field(default_factory=lambda: np.array([6.0, 12.0, 12.0, 1, 1, 1, 1, 1, 1, 1.0]))


class CatchIk:
    """Prioritised damped least-squares IK for the catch.

    Priority 1 puts the cup centre on the ball's path. Priority 2 points the cup along the
    incoming velocity, using only motion that does not move the centre. Priority 3 pulls toward
    a reference posture with what is left. Position comes first because a cup in the right place
    and 30 degrees off still catches, while a well-aimed cup 5 cm away does not, and for catch
    points in front of the chest the aimed pose would put the elbow inside the torso.
    """

    def __init__(self, model: mujoco.MjModel, joint_lo, joint_hi, cfg: IkConfig | None = None):
        self.arm = ArmModel(model)
        self.cfg = cfg or IkConfig()
        self.lo = joint_lo + self.cfg.joint_margin_rad
        self.hi = joint_hi - self.cfg.joint_margin_rad
        self._w_inv = np.diag(1.0 / self.cfg.joint_weights)

    def _pinv(self, j: np.ndarray, damping: float) -> np.ndarray:
        jw = j @ self._w_inv
        return self._w_inv @ j.T @ np.linalg.inv(jw @ j.T + damping**2 * np.eye(j.shape[0]))

    def solve(self, q_seed: np.ndarray, position: np.ndarray, axis: np.ndarray,
              posture: np.ndarray | None = READY_POSTURE, iterations: int | None = None):
        """Returns (q, position error in metres, task Jacobian at q).

        posture=None skips the posture objective. Use it when refining a previous solution for a
        slightly different target: with the posture pull left on, the solution keeps sliding
        through the null space from frame to frame and the reach chases a moving goal.
        """
        cfg = self.cfg
        q = np.clip(q_seed, self.lo, self.hi)
        zero = np.zeros(ARM_DOF)
        eye = np.eye(ARM_DOF)
        total = iterations or cfg.iterations
        # The projectors below use damped inverses, so they are not exact and a little of the
        # posture pull leaks into the task. Near a joint limit that leak can hold the cup 1 to 2 cm
        # off target. The last iterations therefore run without the posture term.
        polish_from = total if posture is None else total
        if posture is not None:
            total += cfg.polish_iterations
        iteration = -1
        while iteration + 1 < total:
            iteration += 1
            if iteration >= polish_from:
                posture = None
            self.arm.update(q, zero)
            j = self.arm.task_jacobian()
            jp, ja = j[:3], j[3:]
            err_p = position - self.arm.cup_position
            err_a = axis_error(self.arm.cup_axes, axis)
            jp_pinv = self._pinv(jp, cfg.position_damping)
            null_p = eye - jp_pinv @ jp
            dq = jp_pinv @ err_p
            ja_null = ja @ null_p
            ja_pinv = self._pinv(ja_null, cfg.tilt_damping)
            dq = dq + null_p @ ja_pinv @ (err_a - ja @ dq)
            null_pa = null_p @ (eye - ja_pinv @ ja_null)
            if posture is not None:
                dq = dq + null_pa @ (cfg.posture_gain * (posture - q))
            peak = np.abs(dq).max()
            if peak > cfg.max_step_rad:
                dq *= cfg.max_step_rad / peak
            q = np.clip(q + dq, self.lo, self.hi)
            if peak < cfg.converged_step_rad:
                if posture is None:
                    break                           # task-only phase has converged
                polish_from = iteration + 1         # posture phase has converged: start the polish
                total = polish_from + cfg.polish_iterations
        self.arm.update(q, zero)
        return q, float(np.linalg.norm(position - self.arm.cup_position)), self.arm.task_jacobian()

    def axis_error_at_solution(self, axis: np.ndarray) -> float:
        """Angle in radians between the cup axis of the last solution and `axis`."""
        return float(np.arccos(np.clip(self.arm.cup_axes[:, 0] @ axis, -1.0, 1.0)))

    def joint_velocity_for(self, task_jacobian: np.ndarray, cup_velocity: np.ndarray) -> np.ndarray:
        """Least-norm joint velocity that moves the cup centre at `cup_velocity` without tilting it."""
        target = np.concatenate([cup_velocity, np.zeros(2)])
        jw = task_jacobian @ self._w_inv
        return self._w_inv @ task_jacobian.T @ np.linalg.solve(jw @ task_jacobian.T + 1e-4 * np.eye(5), target)
