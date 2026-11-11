"""Arm model and the 1 kHz torque laws: computed torque in joint space and operational-space
impedance at the catcher."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from .joints import JointObserver, JointSensingConfig
from .sim import ARM_DOF

CONTROL_PERIOD_S = 0.001
TASK_DIM = 5                     # catcher position (3) and the two tilt angles of its axis
JOINT_FRICTION_NM = 0.3          # Coulomb friction in every joint of the model
FRICTION_SMOOTHING_RAD_S = 0.05  # tanh width used when compensating it


class ArmModel:
    """Rigid-body quantities of the arm, evaluated on a private MjData.

    The ball shares the MuJoCo model but is decoupled from the arm, so the arm's mass matrix and
    bias forces are the leading ARM_DOF block.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self._site = model.site("cup_center").id
        self._body = model.body("catcher").id
        self._full_m = np.zeros((model.nv, model.nv))
        self._bias = np.zeros(model.nv)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))
        self.payload_mass = 0.0   # extra point mass at the cup centre, set by the adaptive estimator

    def update(self, q: np.ndarray, dq: np.ndarray) -> None:
        m, d = self.model, self.data
        d.qpos[:ARM_DOF] = q
        d.qvel[:ARM_DOF] = dq
        d.qvel[ARM_DOF:] = 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_comVel(m, d)

    @property
    def mass_matrix(self) -> np.ndarray:
        mujoco.mj_fullM(self.model, self.data, self._full_m)
        m = self._full_m[:ARM_DOF, :ARM_DOF].copy()
        if self.payload_mass > 0.0:
            jv = self.jacobian()[0]
            m += self.payload_mass * jv.T @ jv
        return m

    @property
    def bias(self) -> np.ndarray:
        """Coriolis, centrifugal and gravity torques, plus the payload's weight if one is set."""
        mujoco.mj_rne(self.model, self.data, 0, self._bias)
        h = self._bias[:ARM_DOF].copy()
        if self.payload_mass > 0.0:
            h += self.payload_mass * 9.81 * self.jacobian()[0][2]
        return h

    def jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear and angular Jacobians of the cup centre, (3, ARM_DOF) each."""
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self._site)
        return self._jp[:, :ARM_DOF].copy(), self._jr[:, :ARM_DOF].copy()

    def jacobian_dot(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_jacDot(self.model, self.data, self._jp, self._jr, self.cup_position, self._body)
        return self._jp[:, :ARM_DOF].copy(), self._jr[:, :ARM_DOF].copy()

    @property
    def cup_position(self) -> np.ndarray:
        return self.data.site_xpos[self._site].copy()

    @property
    def cup_axes(self) -> np.ndarray:
        """Rotation matrix of the catcher. Column 0 is the cup axis, columns 1 and 2 span its rim."""
        return self.data.xmat[self._body].reshape(3, 3).copy()

    def task_jacobian(self) -> np.ndarray:
        """(5, ARM_DOF): cup-centre velocity, then the axis tilt rates about the two rim axes."""
        jp, jr = self.jacobian()
        rim = self.cup_axes[:, 1:]
        return np.vstack([jp, rim.T @ jr])

    def task_jacobian_dot_qdot(self) -> np.ndarray:
        jpd, jrd = self.jacobian_dot()
        dq = self.data.qvel[:ARM_DOF]
        rim = self.cup_axes[:, 1:]
        return np.concatenate([jpd @ dq, rim.T @ (jrd @ dq)])


def axis_error(current_axes: np.ndarray, desired_axis: np.ndarray) -> np.ndarray:
    """Small-angle rotation, about the two rim axes, that takes the cup axis to `desired_axis`."""
    rotation_vector = np.cross(current_axes[:, 0], desired_axis)
    return current_axes[:, 1:].T @ rotation_vector


def lqr_joint_gains(position_weight: float, velocity_weight: float, effort_weight: float = 1.0,
                    period: float = CONTROL_PERIOD_S) -> tuple[float, float]:
    """PD gains (1/s^2, 1/s) for a unit double integrator under zero-order hold, from discrete LQR.

    Computed torque turns every joint into q'' = u, so one gain pair serves all joints.
    """
    a = np.array([[1.0, period], [0.0, 1.0]])
    b = np.array([[0.5 * period**2], [period]])
    q = np.diag([position_weight, velocity_weight])
    r = np.array([[effort_weight]])
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return float(k[0, 0]), float(k[0, 1])


@dataclass
class ImpedanceGains:
    """Physical stiffness of the cup about its reference, in a frame aligned with the approach.

    Damping is set per direction from the task inertia the cup has in that direction, so the
    damping ratios hold whatever the arm's posture.
    """
    along_axis_n_per_m: float = 400.0      # soft where the ball pushes
    lateral_n_per_m: float = 3000.0        # stiff across the ball's path
    tilt_nm_per_rad: float = 40.0
    damping_ratio: float = 1.0
    along_axis_damping_ratio: float = 1.5  # overdamped so the cup absorbs and does not rebound

    def matrices(self, frame: np.ndarray, task_inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stiffness and damping (5x5). frame: columns are the approach axis and two across it."""
        k_local = np.array([self.along_axis_n_per_m, self.lateral_n_per_m, self.lateral_n_per_m])
        zeta = np.array([self.along_axis_damping_ratio, self.damping_ratio, self.damping_ratio])
        inertia_local = np.einsum("ij,ik,kj->j", frame, task_inertia[:3, :3], frame)
        d_local = 2.0 * zeta * np.sqrt(k_local * inertia_local)
        k = np.zeros((TASK_DIM, TASK_DIM))
        d = np.zeros((TASK_DIM, TASK_DIM))
        k[:3, :3] = frame @ np.diag(k_local) @ frame.T
        d[:3, :3] = frame @ np.diag(d_local) @ frame.T
        k[3:, 3:] = self.tilt_nm_per_rad * np.eye(2)
        tilt_inertia = np.diag(task_inertia[3:, 3:])
        d[3:, 3:] = np.diag(2.0 * self.damping_ratio * np.sqrt(self.tilt_nm_per_rad * tilt_inertia))
        return k, d


@dataclass
class ControllerConfig:
    # LQR weights for the joint loop. Chosen for about 20 Hz crossover.
    lqr_position_weight: float = 4.0e7
    lqr_velocity_weight: float = 2.0e3
    posture_stiffness: float = 100.0   # null-space pull toward the reference posture, 1/s^2
    posture_damping: float = 20.0
    task_damping: float = 1e-4         # regularises the task inertia near singularities
    impedance: ImpedanceGains = field(default_factory=ImpedanceGains)
    compensate_friction: bool = True
    sensing: JointSensingConfig = field(default_factory=JointSensingConfig)
    # Robustness to model error, both off by default and compared in scripts/model_error.py.
    cancel_disturbance: bool = False   # subtract the ESO's disturbance estimate (needs observer "eso")
    sliding_gain: float = 0.0          # rad/s^2 of switching action, 0 disables sliding mode
    sliding_slope: float = 30.0        # lambda in s = e' + lambda e, 1/s
    sliding_boundary: float = 0.5      # boundary layer on s, rad/s: saturation in place of sign()


class ArmController:
    """Turns a reference into joint torques. One instance, called every control period.

    Each call reads the encoders through the joint observer, computes a torque, and tells the
    observer which acceleration that torque (after saturation) is expected to produce.
    """

    def __init__(self, arm: ArmModel, torque_limit: np.ndarray, cfg: ControllerConfig | None = None):
        self.arm = arm
        self.cfg = cfg or ControllerConfig()
        if self.cfg.cancel_disturbance and self.cfg.sensing.observer != "eso":
            raise ValueError("cancel_disturbance needs the extended-state observer (sensing.observer = 'eso')")
        self.torque_limit = torque_limit
        self.kp, self.kd = lqr_joint_gains(self.cfg.lqr_position_weight, self.cfg.lqr_velocity_weight)
        self.observer = JointObserver(ARM_DOF, CONTROL_PERIOD_S, self.cfg.sensing)
        self.saturated_ticks = 0
        self.saturated_by_joint = np.zeros(ARM_DOF, dtype=int)

    def reset(self, q: np.ndarray) -> None:
        self.observer.reset(q)
        self.saturated_ticks = 0
        self.saturated_by_joint[:] = 0

    # --- shared pieces -------------------------------------------------------------------------
    def _robust_accel(self, q, dq, q_ref, dq_ref) -> np.ndarray:
        """Acceleration added to reject model error: disturbance cancellation and sliding mode."""
        cfg = self.cfg
        extra = np.zeros(ARM_DOF)
        if cfg.cancel_disturbance:
            extra -= self.observer.disturbance
        if cfg.sliding_gain > 0.0:
            surface = (dq_ref - dq) + cfg.sliding_slope * (q_ref - q)
            extra += cfg.sliding_gain * np.clip(surface / cfg.sliding_boundary, -1.0, 1.0)
        return extra

    def _finish(self, tau_motion: np.ndarray, mass: np.ndarray, bias: np.ndarray, dq_ref: np.ndarray,
                keep_direction: bool) -> np.ndarray:
        """Add bias and friction compensation, respect the torque limits, update the observer.

        keep_direction scales the whole motion torque down when any joint would saturate. Task
        space needs that: clipping joints one by one changes the direction of the commanded
        acceleration and sends the cup somewhere nobody asked for. Joint space does not: the
        joints are decoupled by computed torque, and slowing nine joints because the tenth is
        saturated only makes the reach late.
        """
        base = bias
        if self.cfg.compensate_friction:
            base = base + JOINT_FRICTION_NM * np.tanh(dq_ref / FRICTION_SMOOTHING_RAD_S)
        wanted = base + tau_motion
        over = np.abs(wanted) > self.torque_limit
        if np.any(over):
            self.saturated_ticks += 1
            self.saturated_by_joint += over
        if keep_direction and np.any(over):
            room_hi = self.torque_limit - base
            room_lo = -self.torque_limit - base
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(tau_motion > 0, room_hi / tau_motion,
                                 np.where(tau_motion < 0, room_lo / tau_motion, np.inf))
            wanted = base + float(np.clip(np.min(ratio), 0.0, 1.0)) * tau_motion
        tau = np.clip(wanted, -self.torque_limit, self.torque_limit)
        # What the model says this torque does. Saturation is included, so the observer does not
        # mistake a torque limit for a disturbance.
        self.observer.predict(np.linalg.solve(mass, tau - base))
        return tau

    # --- control laws --------------------------------------------------------------------------
    def computed_torque(self, q_true, dq_true, q_ref, dq_ref, ddq_ref) -> np.ndarray:
        """tau = M (ddq_ref + Kp e + Kd e' + robust terms) + h, on the observer's state estimate."""
        q, dq = self.observer.correct(q_true, dq_true)
        self.arm.update(q, dq)
        accel = ddq_ref + self.kp * (q_ref - q) + self.kd * (dq_ref - dq) + self._robust_accel(q, dq, q_ref, dq_ref)
        mass = self.arm.mass_matrix
        return self._finish(mass @ accel, mass, self.arm.bias, dq_ref, keep_direction=False)

    def impedance(self, q_true, dq_true, x_ref, axis_ref, xd_ref, xdd_ref, q_posture, dq_ref) -> np.ndarray:
        """Operational-space impedance about a moving Cartesian reference:
        tau = J' (Lambda (x_ref'' - J' q') + K e + D e') + N' tau_posture + h.

        x_ref, xd_ref, xdd_ref: cup-centre position, velocity, acceleration. axis_ref: desired cup
        axis. The soft direction of the stiffness is axis_ref.
        """
        arm, cfg = self.arm, self.cfg
        q, dq = self.observer.correct(q_true, dq_true)
        arm.update(q, dq)
        m = arm.mass_matrix
        h = arm.bias
        j = arm.task_jacobian()
        m_inv = np.linalg.inv(m)
        lam = np.linalg.inv(j @ m_inv @ j.T + cfg.task_damping * np.eye(TASK_DIM))
        axes = arm.cup_axes
        # Stiffness frame: first axis along the reference approach direction.
        soft = axis_ref / np.linalg.norm(axis_ref)
        helper = axes[:, 1] - soft * (soft @ axes[:, 1])
        helper /= np.linalg.norm(helper)
        frame = np.column_stack([soft, helper, np.cross(soft, helper)])
        k, d = cfg.impedance.matrices(frame, lam)
        err = np.concatenate([x_ref - arm.cup_position, axis_error(axes, soft)])
        vel = j @ dq
        vel_ref = np.concatenate([xd_ref, np.zeros(2)])
        acc_ref = np.concatenate([xdd_ref, np.zeros(2)])
        # Inertia-weighted feedforward keeps the cup on a moving reference. The spring and damper
        # act as forces, so the stiffness felt at the cup is k itself and not k times the
        # posture-dependent task inertia.
        force = lam @ (acc_ref - arm.task_jacobian_dot_qdot()) + k @ err + d @ (vel_ref - vel)
        tau_task = j.T @ force
        # Posture in the null space of the task, with the dynamically consistent projector.
        j_bar = m_inv @ j.T @ lam
        null = np.eye(ARM_DOF) - j.T @ j_bar.T
        tau_posture = m @ (cfg.posture_stiffness * (q_posture - q) - cfg.posture_damping * dq)
        tau_motion = tau_task + null @ tau_posture
        if cfg.cancel_disturbance:
            tau_motion = tau_motion - m @ self.observer.disturbance
        return self._finish(tau_motion, m, h, dq_ref, keep_direction=True)


class JointTrajectory:
    """Piecewise constant-acceleration joint reference on an absolute clock."""

    def __init__(self, t0: float, q0: np.ndarray, dq0: np.ndarray, accels: np.ndarray, step: float):
        self.t0 = t0
        self.step = step
        self.accels = np.atleast_2d(accels)
        n = len(self.accels)
        self.q = np.empty((n + 1, ARM_DOF))
        self.dq = np.empty((n + 1, ARM_DOF))
        self.q[0], self.dq[0] = q0, dq0
        for k in range(n):
            self.q[k + 1] = self.q[k] + step * self.dq[k] + 0.5 * step**2 * self.accels[k]
            self.dq[k + 1] = self.dq[k] + step * self.accels[k]
        self.t_end = t0 + n * step

    @staticmethod
    def hold(t: float, q: np.ndarray) -> "JointTrajectory":
        return JointTrajectory(t, q, np.zeros(ARM_DOF), np.zeros((1, ARM_DOF)), 1.0)

    def sample(self, t: float):
        """(q, dq, ddq) at time t. Before t0 and after the end the arm is asked to stay still."""
        if t >= self.t_end:
            return self.q[-1], np.zeros(ARM_DOF), np.zeros(ARM_DOF)
        s = max(t - self.t0, 0.0)
        k = min(int(s / self.step), len(self.accels) - 1)
        tau = s - k * self.step
        a = self.accels[k]
        return self.q[k] + tau * self.dq[k] + 0.5 * tau**2 * a, self.dq[k] + tau * a, a
