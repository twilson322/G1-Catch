"""One throw, end to end: cameras, tracker, planner and torque loop on a shared clock.

Three rates run in lockstep with the simulator's 0.5 ms tick:
    torque loop   every 2 ticks   (1 kHz)
    cameras       every 33 ticks  (60.6 Hz)
    planner       once per frame, when the frame leaves the perception pipeline
A frame captured at time t_c becomes available at t_c + perception latency. The tracker files it
at t_c and forecasts from the current time, so the planner always works in the present.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import ballistics as bal
from .control import ArmController, ArmModel, ControllerConfig, JointTrajectory
from .imm import BallTracker, TrackerConfig
from .intercept import CatchIk, Intercept, InterceptConfig, catch_axis, rank_intercepts, refresh_intercept
from .mpc import MpcConfig, ReachMpc
from .sim import ARM_DOF, READY_POSTURE, CatchSim, Throw, model_with_mass_error
from .vision import CameraPose, Intrinsics, camera_poses, in_image, project

CONTROL_EVERY_TICKS = 2
CAMERA_EVERY_TICKS = 33
FORECAST_STEP_S = 0.005
FORECAST_HORIZON_S = 1.0
SELF_COLLISION_CHECK_EVERY_TICKS = 10
MIN_VISIBLE_DEPTH_M = 0.15
ROI_PREDICTION_MARGIN_PX = 40.0   # a predicted pixel this far outside the image still places a search window
NO_COMMIT_GRACE_S = 0.5           # how long past the nominal flight time to keep watching an uncommitted throw
BALL_PARKED_M = (8.0, 8.0, 1.0)   # where the ball is put while checking the robot against itself
GIVE_UP_BEHIND_ROBOT_M = -0.5     # an uncommitted ball this far behind the robot ends the throw
FIRST_IK_ITERATIONS = 80       # the first solve starts far from the answer


@dataclass
class RuntimeConfig:
    perception_latency_s: float = 0.030
    # A plan computed from a frame takes effect this long after the frame arrives. The planner
    # works from where the reference will be at that moment. scripts/planner_timing.py compares
    # the measured compute time with this budget.
    planning_delay_s: float = 0.010
    # What the controller does not know. All zero reproduces an idealised robot.
    model_mass_error: float = 0.0            # each link's mass and inertia wrong by up to this fraction
    model_seed: int = 0
    stereo_vergence_error_deg: float = 0.0   # yaw of the right camera relative to its calibration
    head_pitch_error_deg: float = 0.0        # pitch of both cameras relative to their calibration
    check_self_collision: bool = True        # refuse catch configurations where robot bodies overlap
    compensate_latency: bool = True          # False treats each frame as if it were current
    min_updates_to_commit: int = 4
    max_missed_frames: int = 4               # before commit, this many unused frames in a row restart the track
    max_sigma_to_commit_m: float = 0.30
    replan_every_frame: bool = True          # False plans once at commit
    freeze_before_catch_s: float = 0.012     # no replanning this close to the catch
    impedance_lead_s: float = 0.030          # switch to impedance this long before the catch
    use_impedance: bool = True
    contact_window_s: float = 0.080          # impedance stays on this long after the catch time
    secure_s: float = 0.35                   # then the cup is turned upright over this time
    follow_through_s: float = 0.03           # brake hard: the ball leaves the base within 40 mm
    settle_s: float = 0.60                   # simulated time after the catch before judging it
    large_tracking_error_rad: float = 0.15   # above this the planner restarts from the measured state
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    intercept: InterceptConfig = field(default_factory=InterceptConfig)
    mpc: MpcConfig = field(default_factory=MpcConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass
class ThrowResult:
    caught: bool
    committed: bool
    touched: bool                       # the ball touched the robot at all
    self_collided: bool                 # two robot bodies touched during the throw
    miss_distance_m: float              # closest approach of ball centre to cup centre before any touch
    forecast_error_m: float             # final planned catch point vs the ball's true position then
    cup_error_m: float                  # cup centre vs the planned catch point at the catch time, across the ball's path
    reference_error_m: float            # same, for the reference trajectory: what the planner asked for
    aim_error_rad: float                # cup axis vs the reversed ball velocity at the catch time
    catch_time: float                   # planned, nan if never committed
    commit_time: float
    frames_used: int
    relative_speed_at_contact: float    # ball speed relative to the cup at first touch, m/s
    peak_torque_fraction: float
    saturated_ticks: int
    saturated_ticks_reach: int          # control periods with a torque limit active before the contact phase
    reach_tracking_error_rad: float     # largest joint error against the reference during the reach
    saturated_joint: int                # index of the joint that saturated most often, -1 if none
    planner_over_budget: int            # planner calls that took longer than the planning delay
    planner_ms: list
    frames: list                        # per-frame log, see CatchRunner._log_frame
    commit_features: dict | None


@dataclass
class _ThrowState:
    """Everything that changes during one throw. Kept apart from CatchRunner so a runner holds no
    state between throws."""
    tracker: BallTracker
    traj: JointTrajectory
    posture: np.ndarray
    q_goal_seed: np.ndarray
    q_final: np.ndarray
    catch_axis: np.ndarray
    end_time: float
    plan: Intercept | None = None
    adjust: tuple | None = None
    previous_accel: np.ndarray = field(default_factory=lambda: np.zeros(ARM_DOF))
    commit_time: float = float("nan")
    commit_features: dict | None = None
    refused: bool = False
    secured: bool = False
    # Plans that have been computed and not yet taken effect: (start time, trajectory, final posture).
    # More than one can be waiting when the planning delay is longer than a camera frame.
    pending: deque = field(default_factory=deque)
    saturated_reach: int = 0
    reach_tracking_error: float = 0.0
    over_budget: int = 0
    missed_frames: int = 0
    frames: list = field(default_factory=list)
    planner_ms: list = field(default_factory=list)
    # Scoring against simulator truth. Never fed back to the controller.
    miss: float = float("inf")
    touched: bool = False
    self_collided: bool = False
    forecast_error: float = float("nan")
    cup_error: float = float("nan")
    reference_error: float = float("nan")
    aim_error: float = float("nan")
    rel_speed_at_contact: float = float("nan")
    peak_torque: float = 0.0


class CatchRunner:
    def __init__(self, sim: CatchSim, sensor, cfg: RuntimeConfig | None = None, residual=None,
                 reach_planner=None, intercept_adjust=None, catch_gate=None, kd: float | None = None):
        """
        residual: learned acceleration correction used by the tracker and the forecast.
        kd: drag constant of the tracker's flight model, 1/m. None uses the nominal sphere.
        reach_planner: object with the ReachMpc.solve interface (a learned policy plugs in here).
        intercept_adjust: callable(features) -> (dt, dp) correction to the chosen intercept.
        catch_gate: callable(features) -> bool, False refuses to commit to the throw.
        """
        self.sim = sim
        self.sensor = sensor
        self.cfg = cfg or RuntimeConfig()
        self.intr = Intrinsics.from_fovy(sim.model.cam_fovy[0])
        self.residual = residual
        self.kd = bal.NOMINAL_KD if kd is None else kd
        # The controller's dynamics model may have wrong link masses. Kinematics are shared.
        self.arm = ArmModel(model_with_mass_error(sim.model, self.cfg.model_mass_error,
                                                  np.random.default_rng(self.cfg.model_seed)))
        self.collision_data = mujoco.MjData(sim.model)
        self.ref_arm = ArmModel(sim.model)
        self.controller = ArmController(self.arm, sim.torque_limit, self.cfg.controller)
        self.ik = CatchIk(sim.model, sim.joint_lo, sim.joint_hi)
        self.planner = reach_planner or ReachMpc(sim.joint_lo, sim.joint_hi, self.cfg.mpc)
        self.secure_mpc = ReachMpc(sim.joint_lo, sim.joint_hi, self.cfg.mpc)
        self.intercept_adjust = intercept_adjust
        self.catch_gate = catch_gate

    def _new_tracker(self) -> BallTracker:
        return BallTracker(self.intr, self.cfg.tracker, kd=self.kd, residual=self.residual)

    # --- one throw ---------------------------------------------------------------------------
    def run(self, throw: Throw, posture: np.ndarray = READY_POSTURE) -> ThrowResult:
        sim, cfg = self.sim, self.cfg
        sim.reset(throw, posture)
        self.controller.reset(posture)
        st = _ThrowState(tracker=self._new_tracker(), traj=JointTrajectory.hold(0.0, posture.copy()),
                         posture=posture.copy(), q_goal_seed=posture.copy(), q_final=posture.copy(),
                         catch_axis=sim.cup_axis,
                         end_time=throw.flight_time + NO_COMMIT_GRACE_S + cfg.settle_s)
        pending = deque()                      # frames inside the perception pipeline
        latency_ticks = int(round(cfg.perception_latency_s / sim.dt))
        tau = np.zeros(ARM_DOF)
        tick = 0
        while sim.time < st.end_time:
            now = sim.time
            frames_still_useful = st.plan is None or now < st.plan.time
            if tick % CAMERA_EVERY_TICKS == 0 and frames_still_useful:
                try:
                    frame = self._capture(st, now)
                except np.linalg.LinAlgError:
                    st.tracker = self._new_tracker()
                    frame = self._capture(st, now)
                pending.append((tick + latency_ticks, now, frame))
            if pending and pending[0][0] <= tick:
                _, t_capture, meas = pending.popleft()
                try:
                    self._perceive(st, meas, t_capture, now)
                    self._plan(st, now)
                    self._log_frame(st, now, t_capture, meas)
                except np.linalg.LinAlgError:
                    # The estimate has become numerically meaningless (seen only with the outlier
                    # gate disabled). Drop the track and start again from the next frame.
                    st.tracker = self._new_tracker()
            if tick % CONTROL_EVERY_TICKS == 0:
                tau = self._torque(st, now)
                st.peak_torque = max(st.peak_torque, float(np.max(np.abs(tau) / sim.torque_limit)))
            sim.step(tau)
            tick += 1
            self._score(st, tick)
            if st.plan is None and sim.ball_position[0] < GIVE_UP_BEHIND_ROBOT_M:
                break

        return ThrowResult(
            caught=sim.ball_in_cup(), committed=st.plan is not None, touched=st.touched,
            self_collided=st.self_collided, miss_distance_m=float(st.miss),
            forecast_error_m=st.forecast_error, cup_error_m=st.cup_error, reference_error_m=st.reference_error,
            aim_error_rad=st.aim_error, catch_time=st.plan.time if st.plan else float("nan"),
            commit_time=st.commit_time, frames_used=st.tracker.updates,
            relative_speed_at_contact=st.rel_speed_at_contact, peak_torque_fraction=st.peak_torque,
            saturated_ticks=self.controller.saturated_ticks, saturated_ticks_reach=int(st.saturated_reach),
            saturated_joint=int(np.argmax(self.controller.saturated_by_joint)) if self.controller.saturated_ticks else -1,
            reach_tracking_error_rad=st.reach_tracking_error, planner_over_budget=int(st.over_budget), planner_ms=st.planner_ms, frames=st.frames,
            commit_features=st.commit_features)

    # --- cameras and tracker -----------------------------------------------------------------
    def _capture(self, st: _ThrowState, now: float):
        """Take a stereo frame now. With a track, tell the detector where to look."""
        predicted = None
        if st.tracker.initialised:
            _, p_pred, _, _ = st.tracker.forecast(now, 1e-6, 1)
            poses = camera_poses(self.sim.model, self.sim.data)
            predicted = []
            for pose in poses:
                px, depth = project(self.intr, pose, p_pred[0])
                near_image = depth > MIN_VISIBLE_DEPTH_M and in_image(self.intr, px, -ROI_PREDICTION_MARGIN_PX)
                predicted.append(px if near_image else None)
        meas = self.sensor.capture(self.sim, predicted)
        meas.poses = self._miscalibrated(meas.poses)
        return meas

    def _miscalibrated(self, poses):
        """The camera poses the tracker believes, which differ from the true ones by the configured
        calibration errors. The images come from the true poses."""
        cfg = self.cfg
        if cfg.stereo_vergence_error_deg == 0.0 and cfg.head_pitch_error_deg == 0.0:
            return poses
        out = []
        for index, pose in enumerate(poses):
            rotation = pose.rotation
            pitch = np.radians(cfg.head_pitch_error_deg)
            if pitch:
                c, s_ = np.cos(pitch), np.sin(pitch)
                rotation = rotation @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])   # about camera x
            yaw = np.radians(cfg.stereo_vergence_error_deg) if index == 1 else 0.0
            if yaw:
                c, s_ = np.cos(yaw), np.sin(yaw)
                rotation = rotation @ np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])   # about camera y
            out.append(CameraPose(pose.position, rotation))
        return tuple(out)

    def _perceive(self, st: _ThrowState, meas, t_capture: float, now: float) -> None:
        """File a frame that has just left the perception pipeline."""
        stamp = t_capture if self.cfg.compensate_latency else now
        if not st.tracker.initialised:
            st.tracker.initialise(meas, stamp)
        elif st.tracker.step(meas, stamp):
            st.missed_frames = 0
        else:
            st.missed_frames += 1
            if st.plan is None and st.missed_frames >= self.cfg.max_missed_frames:
                # The track was started on something that was not the ball. Start over.
                st.tracker = self._new_tracker()
                st.missed_frames = 0

    # --- planning ----------------------------------------------------------------------------
    def _plan(self, st: _ThrowState, now: float) -> None:
        """Choose or refresh the catch point and re-plan the reach to it."""
        cfg = self.cfg
        ready = st.tracker.initialised and st.tracker.updates >= cfg.min_updates_to_commit and not st.refused
        if st.plan is not None:
            ready = ready and cfg.replan_every_frame \
                and now + cfg.planning_delay_s < st.plan.time - cfg.freeze_before_catch_s
        if not ready:
            return
        t_wall = time.perf_counter()
        # The plan takes effect after the planning delay, so it starts from where the reference
        # that is being tracked now will be at that moment.
        start = now + cfg.planning_delay_s
        active = st.pending[-1][1] if st.pending else st.traj     # the newest plan is the one in force at `start`
        q_ref, dq_ref, _ = active.sample(start)
        if np.abs(q_ref - self.controller.observer.q).max() > cfg.large_tracking_error_rad:
            q_ref, dq_ref = self.controller.observer.q.copy(), self.controller.observer.dq.copy()
        self.ref_arm.update(q_ref, np.zeros(ARM_DOF))

        target = self._refresh(st, now) if st.plan is not None else None
        if target is None:
            target = self._choose(st, now)
        if target is not None:
            self._commit_to(st, target, start, q_ref, dq_ref)
        elapsed = time.perf_counter() - t_wall
        st.planner_ms.append(1e3 * elapsed)
        st.over_budget += elapsed > cfg.planning_delay_s

    def _refresh(self, st: _ThrowState, now: float) -> Intercept | None:
        """The committed catch time with the newest forecast, or None if the ball is no longer
        forecast to be reachable at that time."""
        _, p, v, cov = st.tracker.forecast(now, st.plan.time - now, 1)
        return refresh_intercept(st.plan, p[0], v[0], cov[0], now, self.cfg.intercept)

    def _choose(self, st: _ThrowState, now: float) -> Intercept | None:
        """The best forecast point that inverse kinematics can reach, subject to the catch gate.

        Before commit, a point the arm cannot reach is skipped and the throw waits for a better
        forecast. After commit the arm is already moving, so if no candidate passes the IK check
        the closest one is taken as a best effort.
        """
        cfg = self.cfg
        steps = int(FORECAST_HORIZON_S / FORECAST_STEP_S)
        times, ps, vs, covs = st.tracker.forecast(now, FORECAST_STEP_S, steps)
        first_commit = st.plan is None
        options = rank_intercepts(times, ps, vs, covs, now, self.ref_arm.cup_position, cfg.intercept,
                                  near_time=None if first_commit else st.plan.time)
        best_effort = None
        for option in options:
            if first_commit and option.position_sigma >= cfg.max_sigma_to_commit_m:
                continue
            axis = catch_axis(option.ball_velocity, cfg.intercept.min_axis_up)
            q_goal, ik_error, _ = self.ik.solve(st.posture, option.position, axis, iterations=FIRST_IK_ITERATIONS)
            if ik_error < cfg.intercept.max_ik_error_m and cfg.check_self_collision and self._collides(q_goal):
                continue                           # reachable only with the arm inside the body
            if ik_error >= cfg.intercept.max_ik_error_m:
                if best_effort is None or ik_error < best_effort[0]:
                    best_effort = (ik_error, option, q_goal)
                continue
            if first_commit:
                st.commit_features = self._features(option, now)
                if self.catch_gate is not None and not self.catch_gate(st.commit_features):
                    st.refused = True
                    return None
                if self.intercept_adjust is not None:
                    st.adjust = self.intercept_adjust(st.commit_features)
                    option.time += st.adjust[0]      # the time shift is applied once, here
            st.q_goal_seed = q_goal
            return option
        if not first_commit and best_effort is not None:
            st.q_goal_seed = best_effort[2]
            return best_effort[1]
        return None

    def _commit_to(self, st: _ThrowState, target: Intercept, start: float, q_ref, dq_ref) -> None:
        """Plan the reach to `target` as a trajectory that takes over at time `start`."""
        cfg = self.cfg
        if st.adjust is not None:
            # The position offset applies to every new forecast of the ball at the catch time.
            target.position = target.position + st.adjust[1]
        if st.plan is None:
            st.commit_time = start
        st.plan = target
        speed = np.linalg.norm(target.ball_velocity)
        direction = target.ball_velocity / speed
        st.catch_axis = catch_axis(target.ball_velocity, cfg.intercept.min_axis_up)
        # Refine the previous goal for the new target. No posture pull here, so the goal does
        # not slide through the null space between frames.
        q_goal, _, jac = self.ik.solve(st.q_goal_seed, target.position, st.catch_axis, posture=None)
        st.q_goal_seed = q_goal
        cup_speed = min(cfg.intercept.velocity_match * speed, cfg.intercept.max_cup_speed_m_s)
        dq_goal = self.ik.joint_velocity_for(jac, cup_speed * direction)     # retreat along the ball's path
        sol = self.planner.solve(q_ref, dq_ref, q_goal, dq_goal, max(target.time - start, 1e-3), st.previous_accel)
        st.pending.append((start, *self._with_follow_through(start, q_ref, dq_ref, sol)))
        st.previous_accel = sol.accels[0]
        st.end_time = target.time + cfg.settle_s

    def _collides(self, q: np.ndarray) -> bool:
        """True if two robot bodies overlap in configuration q, by MuJoCo's collision meshes."""
        m, d = self.sim.model, self.collision_data
        d.qpos[:ARM_DOF] = q
        d.qpos[ARM_DOF:ARM_DOF + 3] = BALL_PARKED_M
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0 and m.geom_bodyid[c.geom1] != 0 and m.geom_bodyid[c.geom2] != 0:
                return True
        return False

    def _with_follow_through(self, now, q0, dq0, sol):
        """Append a constant deceleration to rest after the catch."""
        reach = JointTrajectory(now, q0, dq0, sol.accels, sol.step)
        n = max(1, int(np.ceil(self.cfg.follow_through_s / sol.step)))
        brake = np.tile(-reach.dq[-1] / (n * sol.step), (n, 1))
        full = JointTrajectory(now, q0, dq0, np.vstack([sol.accels, brake]), sol.step)
        return full, full.q[-1].copy()

    def _secure_trajectory(self, now, q, dq) -> JointTrajectory:
        """From the measured state to rest with the cup opening straight up, where it is now."""
        self.ref_arm.update(q, np.zeros(ARM_DOF))
        q_up, _, _ = self.ik.solve(q, self.ref_arm.cup_position, np.array([0.0, 0.0, 1.0]))
        sol = self.secure_mpc.solve(q, dq, q_up, np.zeros(ARM_DOF), self.cfg.secure_s)
        return JointTrajectory(now, q, dq, sol.accels, sol.step)

    # --- torque loop -------------------------------------------------------------------------
    def _torque(self, st: _ThrowState, now: float) -> np.ndarray:
        """One control period. Reach: computed torque. Around the catch: impedance. Then secure."""
        cfg = self.cfg
        q, dq = self.sim.q, self.sim.dq
        while st.pending and now >= st.pending[0][0]:
            _, st.traj, st.q_final = st.pending.popleft()
        if st.plan is not None and not st.secured and now >= st.plan.time + cfg.contact_window_s:
            # The ball is in the cup or gone. Turn the cup upright so a caught ball stays.
            observer = self.controller.observer
            st.traj = self._secure_trajectory(now, observer.q.copy(), observer.dq.copy())
            st.pending.clear()
            st.secured = True
        q_ref, dq_ref, ddq_ref = st.traj.sample(now)
        contact_phase = st.plan is not None and cfg.use_impedance and not st.secured \
            and now >= st.plan.time - cfg.impedance_lead_s
        before = self.controller.saturated_ticks
        if not contact_phase:
            tau = self.controller.computed_torque(q, dq, q_ref, dq_ref, ddq_ref)
        else:
            self.ref_arm.update(q_ref, dq_ref)
            jp, _ = self.ref_arm.jacobian()
            jpd, _ = self.ref_arm.jacobian_dot()
            tau = self.controller.impedance(q, dq, self.ref_arm.cup_position, st.catch_axis, jp @ dq_ref,
                                            jp @ ddq_ref + jpd @ dq_ref, st.q_final, dq_ref)
        reaching = st.plan is not None and now < st.plan.time - cfg.impedance_lead_s
        st.saturated_reach += reaching and self.controller.saturated_ticks > before
        if reaching:      # truth-based score, not fed back
            st.reach_tracking_error = max(st.reach_tracking_error, float(np.abs(q_ref - q).max()))
        return tau

    # --- scoring and logs --------------------------------------------------------------------
    def _score(self, st: _ThrowState, tick: int) -> None:
        sim = self.sim
        if not st.touched:
            st.miss = min(st.miss, float(np.linalg.norm(sim.ball_position - sim.cup_position)))
            if sim.ball_touching_robot():
                st.touched = True
                st.rel_speed_at_contact = float(np.linalg.norm(sim.ball_velocity - sim.cup_velocity()))
        if st.plan is not None and np.isnan(st.cup_error) and sim.time >= st.plan.time:
            across = lambda offset: float(np.linalg.norm(offset - (offset @ st.catch_axis) * st.catch_axis))
            st.cup_error = across(sim.cup_position - st.plan.position)
            st.aim_error = float(np.arccos(np.clip(sim.cup_axis @ st.catch_axis, -1.0, 1.0)))
            self.ref_arm.update(st.traj.sample(st.plan.time)[0], np.zeros(ARM_DOF))
            st.reference_error = across(self.ref_arm.cup_position - st.plan.position)
            if not st.touched:
                st.forecast_error = float(np.linalg.norm(sim.ball_position - st.plan.position))
        if tick % SELF_COLLISION_CHECK_EVERY_TICKS == 0 and not st.self_collided and sim.self_collision():
            st.self_collided = True

    def _features(self, cand: Intercept, now: float) -> dict:
        """What the planner knows about a catch point when it commits. Inputs to the learned gate
        and to the learned intercept correction."""
        rel = cand.position - self.cfg.intercept.shoulder
        speed = float(np.linalg.norm(cand.ball_velocity))
        return {
            "time_to_go": cand.time - now, "slack": cand.slack_s, "reach": float(np.linalg.norm(rel)),
            "rel_x": float(rel[0]), "rel_y": float(rel[1]), "rel_z": float(rel[2]),
            "sigma": cand.position_sigma, "speed": speed, "descent": float(-cand.ball_velocity[2] / speed),
            "travel": float(np.linalg.norm(cand.position - self.ref_arm.cup_position)),
            "vx": float(cand.ball_velocity[0]), "vy": float(cand.ball_velocity[1]), "vz": float(cand.ball_velocity[2]),
        }

    def _log_frame(self, st: _ThrowState, now: float, t_capture: float, meas) -> None:
        tracker = st.tracker
        if not tracker.initialised:
            return
        sim = self.sim
        # Truth at the capture time is not stored, so the estimate is advanced to now and
        # compared with the current truth.
        _, p_now, v_now, cov_now = tracker.forecast(now, 1e-6, 1)
        plan = st.plan
        st.frames.append({
            "t": now, "t_capture": t_capture, "cameras": len(meas.cameras()),
            "est_p": p_now[0], "est_v": v_now[0], "sigma_now": float(np.sqrt(np.trace(cov_now[0]))),
            "true_p": sim.ball_position, "true_v": sim.ball_velocity,
            "mode_prob": tracker.mu.copy(), "nis": tracker.last_nis,
            "plan_t": plan.time if plan else float("nan"),
            "plan_p": plan.position.copy() if plan else np.full(3, np.nan),
            "plan_sigma": plan.position_sigma if plan else float("nan"),
        })
