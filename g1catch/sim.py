"""The simulated world: G1 upper body, a thrown ball with aerodynamics, and throw sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import MODEL_PATH
from . import ballistics as bal

ARM_DOF = 10                       # 3 waist + 7 right arm joints, torque controlled
# Ready posture: forearm forward and up, cup opening toward the thrower.
READY_POSTURE = np.array([0.0, 0.0, 0.0, -0.6, -0.15, 0.0, -0.3, 0.0, 0.0, 0.0])
# The upper arm enters the torso's collision mesh when the shoulder rolls inward past about
# zero (scripts/self_collision_map.py). Planning keeps it outside with this limit.
SHOULDER_ROLL_INDEX = 4
SHOULDER_ROLL_PLANNING_MAX_RAD = -0.05
# Caught means the ball's centre is inside the cup's volume and moving with it at the end of the
# episode. Distances are measured from the cup_center site, which sits 34 mm above the base.
CAUGHT_AXIAL_RANGE_M = (-0.04, 0.05)
CAUGHT_MAX_RADIAL_M = 0.075          # rim radius
CAUGHT_MAX_RELATIVE_SPEED_M_S = 0.5
MAX_BOUNCE_THROW_SPEED_M_S = 8.0
MIN_TIME_AFTER_BOUNCE_S = 0.22


@dataclass
class Throw:
    position: np.ndarray            # release point, world frame, m
    velocity: np.ndarray            # release velocity, m/s
    spin: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s
    drag_coefficient: float = bal.NOMINAL_DRAG_COEFFICIENT
    mass: float = bal.BALL_MASS_KG
    aim: np.ndarray | None = None   # drag-free target point the throw was built for
    flight_time: float = 0.0        # drag-free time to reach `aim`


@dataclass
class ThrowDistribution:
    """Direct throws aimed at a box in front of the right shoulder.

    The box stops 8 cm short of the body's midline. Further left the upper arm would have to pass
    through the torso (see WIDE_AIM for throws that include that region).
    """
    release_lo: tuple = (2.8, -0.8, 0.9)
    release_hi: tuple = (3.8, 0.8, 1.6)
    aim_lo: tuple = (0.26, -0.42, 0.95)
    aim_hi: tuple = (0.46, -0.02, 1.40)
    flight_time_s: tuple = (0.55, 0.85)
    drag_coefficient: tuple = (0.40, 0.60)
    max_spin_rad_s: float = 25.0
    bounce: bool = False            # aim the drag-free arc at the floor first

    def sample(self, rng: np.random.Generator) -> Throw:
        cd = float(rng.uniform(*self.drag_coefficient))
        spin = rng.uniform(-self.max_spin_rad_s, self.max_spin_rad_s, 3)
        if self.bounce:
            return self._sample_bounce(rng, spin, cd)
        p0 = rng.uniform(self.release_lo, self.release_hi)
        aim = rng.uniform(self.aim_lo, self.aim_hi)
        t = float(rng.uniform(*self.flight_time_s))
        v0 = (aim - p0) / t + 0.5 * np.array([0.0, 0.0, bal.GRAVITY_M_S2]) * t
        return Throw(p0, v0, spin, cd, aim=aim, flight_time=t)

    def _sample_bounce(self, rng, spin, cd) -> Throw:
        """A throw that hits the floor once and then crosses the aim box.

        Built by rejection sampling with the nominal bounce model and no drag. About one draw in
        seven is accepted. A rebound to catch height needs a hard downward throw, so these
        arrive faster and with less warning than direct throws.
        """
        r, g = bal.BALL_RADIUS_M, bal.GRAVITY_M_S2
        for _ in range(10_000):
            p0 = rng.uniform(self.release_lo, self.release_hi)
            aim = rng.uniform(self.aim_lo, self.aim_hi)
            frac = rng.uniform(0.40, 0.70)             # bounce point along the ground track
            t1 = rng.uniform(0.15, 0.40)               # time to the bounce
            bounce_xy = p0[:2] + frac * (aim[:2] - p0[:2])
            v_xy = (bounce_xy - p0[:2]) / t1
            if np.linalg.norm(v_xy) > MAX_BOUNCE_THROW_SPEED_M_S:
                continue
            vz0 = (r - p0[2]) / t1 + 0.5 * g * t1
            vz_out = -bal.NOMINAL_RESTITUTION * (vz0 - g * t1)
            v_xy_out = bal.NOMINAL_TANGENTIAL_RETENTION * v_xy
            t2 = np.linalg.norm(aim[:2] - bounce_xy) / np.linalg.norm(v_xy_out)
            z_at_aim = r + vz_out * t2 - 0.5 * g * t2**2
            if self.aim_lo[2] < z_at_aim < self.aim_hi[2] and t2 > MIN_TIME_AFTER_BOUNCE_S:
                aim = np.array([aim[0], aim[1], z_at_aim])
                return Throw(p0, np.array([v_xy[0], v_xy[1], vz0]), spin, cd, aim=aim, flight_time=t1 + t2)
        raise RuntimeError("no bounce throw satisfies the release and aim boxes")


# A larger box that includes points the arm cannot reach. Used to train and test the
# catchability classifier.
WIDE_AIM_LO = (0.15, -0.70, 0.75)
WIDE_AIM_HI = (0.60, 0.30, 1.60)


def model_with_mass_error(model: mujoco.MjModel, error: float, rng: np.random.Generator) -> mujoco.MjModel:
    """A copy of `model` whose moving robot links have wrong masses, for the controller to use.

    Each link's mass and inertia are scaled by one factor drawn uniformly from 1 +/- error, which
    is what a CAD-derived model with unmodelled cables, covers and fasteners looks like. Kinematics
    are unchanged.
    """
    import copy
    wrong = copy.copy(model)
    if error <= 0.0:
        return wrong
    ball = model.body("ball").id
    for body in range(1, model.nbody):
        if body == ball or model.body_dofnum[body] == 0 and model.body_weldid[body] == 0:
            continue
        factor = 1.0 + rng.uniform(-error, error)
        wrong.body_mass[body] = model.body_mass[body] * factor
        wrong.body_inertia[body] = model.body_inertia[body] * factor
    return wrong


def load_arm_only(timestep: float | None = None) -> mujoco.MjModel:
    """The same robot with no ball, for trajectory optimisation over the arm's own dynamics."""
    import re
    xml = Path(MODEL_PATH).read_text()
    xml = re.sub(r'<body name="ball".*?</body>', "", xml, flags=re.S)
    xml = xml.replace('<framepos name="ball_pos" objtype="body" objname="ball"/>', "")
    xml = xml.replace('meshdir="assets"', f'meshdir="{Path(MODEL_PATH).parent / "assets"}"')
    model = mujoco.MjModel.from_xml_string(xml)
    if timestep is not None:
        model.opt.timestep = timestep
    return model


class CatchSim:
    def __init__(self, model_path=MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self._ball_body = self.model.body("ball").id
        self._cup_site = self.model.site("cup_center").id
        self._catcher_body = self.model.body("catcher").id
        self._ball_geom = self.model.geom("ball_geom").id
        self._cup_geoms = [g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith("cup_")]
        self._nominal_ball_inertia = self.model.body_inertia[self._ball_body].copy()
        self.torque_limit = self.model.actuator_ctrlrange[:, 1].copy()
        self.joint_lo = self.model.jnt_range[:ARM_DOF, 0].copy()
        self.joint_hi = self.model.jnt_range[:ARM_DOF, 1].copy()
        self.joint_hi[SHOULDER_ROLL_INDEX] = SHOULDER_ROLL_PLANNING_MAX_RAD
        self._kd = bal.NOMINAL_KD
        self._km = bal.magnus_per_radian()
        self._mass = bal.BALL_MASS_KG

    # --- setup -------------------------------------------------------------------------------
    def reset(self, throw: Throw, posture: np.ndarray = READY_POSTURE) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        scale = throw.mass / bal.BALL_MASS_KG
        m.body_mass[self._ball_body] = throw.mass
        m.body_inertia[self._ball_body] = self._nominal_ball_inertia * scale
        if throw.mass != self._mass:
            # The contact solver scales its impedance with inverse weights derived from the masses.
            # Leaving them stale after a mass change makes a heavier ball's contacts unstable.
            mujoco.mj_setConst(m, d)
        self._mass = throw.mass
        self._kd = bal.drag_per_metre(throw.drag_coefficient, mass=throw.mass)
        self._km = bal.magnus_per_radian(mass=throw.mass)
        d.qpos[:ARM_DOF] = posture
        d.qpos[ARM_DOF:ARM_DOF + 3] = throw.position
        d.qpos[ARM_DOF + 3:ARM_DOF + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[ARM_DOF:ARM_DOF + 3] = throw.velocity
        d.qvel[ARM_DOF + 3:ARM_DOF + 6] = throw.spin   # identity orientation: body = world axes
        mujoco.mj_forward(m, d)

    def set_cup_restitution(self, restitution: float) -> None:
        """Change how much relative speed the cup's liner returns, keeping its stiffness.

        For a linear spring-damper contact the restitution is exp(-zeta pi / sqrt(1 - zeta^2)).
        """
        stiffness = -self.model.geom_solref[self._cup_geoms[0], 0]
        log_e = np.log(np.clip(restitution, 1e-3, 0.999))
        zeta = -log_e / np.sqrt(np.pi**2 + log_e**2)
        self.model.geom_solref[self._cup_geoms, 1] = -2.0 * zeta * np.sqrt(stiffness)

    # --- stepping ----------------------------------------------------------------------------
    def step(self, torque: np.ndarray) -> None:
        d = self.data
        d.ctrl[:] = np.clip(torque, -self.torque_limit, self.torque_limit)
        v = d.qvel[ARM_DOF:ARM_DOF + 3]
        omega_world = d.xmat[self._ball_body].reshape(3, 3) @ d.qvel[ARM_DOF + 3:ARM_DOF + 6]
        accel = -self._kd * np.linalg.norm(v) * v + self._km * np.cross(omega_world, v)
        d.xfrc_applied[self._ball_body, :3] = self._mass * accel
        mujoco.mj_step(self.model, d)

    # --- accessors ---------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return self.data.time

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[:ARM_DOF].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[:ARM_DOF].copy()

    @property
    def ball_position(self) -> np.ndarray:
        return self.data.qpos[ARM_DOF:ARM_DOF + 3].copy()

    @property
    def ball_velocity(self) -> np.ndarray:
        return self.data.qvel[ARM_DOF:ARM_DOF + 3].copy()

    @property
    def cup_position(self) -> np.ndarray:
        return self.data.site_xpos[self._cup_site].copy()

    @property
    def cup_axis(self) -> np.ndarray:
        """Unit vector out of the cup opening."""
        return self.data.xmat[self._catcher_body].reshape(3, 3)[:, 0].copy()

    def cup_velocity(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self._cup_site)
        return jac @ self.data.qvel

    def self_collision(self) -> bool:
        """True if two robot bodies are in contact (the ball and the floor do not count)."""
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0 and self._ball_geom not in (c.geom1, c.geom2) \
                    and self.model.geom_bodyid[c.geom1] != 0 and self.model.geom_bodyid[c.geom2] != 0:
                return True
        return False

    def ball_touching_robot(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if self._ball_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._ball_geom else c.geom1
                if self.model.geom_bodyid[other] != 0:
                    return True
        return False

    def ball_in_cup(self) -> bool:
        rel = self.ball_position - self.cup_position
        axis = self.cup_axis
        axial = float(rel @ axis)
        radial = float(np.linalg.norm(rel - axial * axis))
        rel_speed = np.linalg.norm(self.ball_velocity - self.cup_velocity())
        return bool(CAUGHT_AXIAL_RANGE_M[0] < axial < CAUGHT_AXIAL_RANGE_M[1] and radial < CAUGHT_MAX_RADIAL_M
                    and rel_speed < CAUGHT_MAX_RELATIVE_SPEED_M_S)
