import mujoco
import numpy as np
import pytest
from scipy.optimize import lsq_linear

from g1catch import ilqr
from g1catch.adaptive import PayloadEstimator, ScalarRls
from g1catch.control import (ArmController, ArmModel, ControllerConfig, ImpedanceGains, JointTrajectory, lqr_joint_gains)
from g1catch.intercept import CatchIk, InterceptConfig, choose_intercept, refresh_intercept
from g1catch.mpc import MpcConfig, ReachMpc, prediction_matrices
from g1catch.sim import ARM_DOF, READY_POSTURE, CatchSim, Throw, ThrowDistribution


@pytest.fixture(scope="module")
def sim():
    s = CatchSim()
    s.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3)))
    return s


def test_arm_model_matches_mujoco_inverse_dynamics(sim):
    """M qdd + h from ArmModel equals MuJoCo's own inverse dynamics for random states."""
    rng = np.random.default_rng(0)
    arm = ArmModel(sim.model)
    data = mujoco.MjData(sim.model)
    for _ in range(10):
        q = rng.uniform(sim.joint_lo, sim.joint_hi)
        dq = rng.normal(0, 2.0, ARM_DOF)
        ddq = rng.normal(0, 20.0, ARM_DOF)
        arm.update(q, dq)
        tau = arm.mass_matrix @ ddq + arm.bias
        data.qpos[:ARM_DOF], data.qvel[:ARM_DOF], data.qacc[:] = q, dq, 0.0
        data.qacc[:ARM_DOF] = ddq
        mujoco.mj_inverse(sim.model, data)
        # qfrc_inverse is the applied force, which excludes passive forces and joint friction
        # (a constraint force in MuJoCo). Adding them back leaves the rigid-body torque.
        expected = (data.qfrc_inverse + data.qfrc_passive + data.qfrc_constraint)[:ARM_DOF]
        assert tau == pytest.approx(expected, abs=1e-8)


def test_task_jacobian_matches_finite_differences(sim):
    arm = ArmModel(sim.model)
    q = READY_POSTURE + 0.2
    arm.update(q, np.zeros(ARM_DOF))
    jac = arm.task_jacobian()
    p0, axes0 = arm.cup_position, arm.cup_axes
    eps = 1e-6
    for j in range(ARM_DOF):
        dq = np.zeros(ARM_DOF)
        dq[j] = eps
        arm.update(q + dq, np.zeros(ARM_DOF))
        assert (arm.cup_position - p0) / eps == pytest.approx(jac[:3, j], abs=1e-5)
        tilt = axes0[:, 1:].T @ np.cross(axes0[:, 0], arm.cup_axes[:, 0])
        assert tilt / eps == pytest.approx(jac[3:, j], abs=1e-5)


def test_payload_enters_mass_matrix_and_gravity(sim):
    arm = ArmModel(sim.model)
    arm.update(READY_POSTURE, np.zeros(ARM_DOF))
    m0, h0 = arm.mass_matrix, arm.bias
    arm.payload_mass = 0.4
    jv = arm.jacobian()[0]
    assert arm.mass_matrix - m0 == pytest.approx(0.4 * jv.T @ jv)
    assert arm.bias - h0 == pytest.approx(0.4 * 9.81 * jv[2])


def test_lqr_gains_solve_the_riccati_equation_and_stabilise():
    kp, kd = lqr_joint_gains(4.0e7, 2.0e3)
    t = 0.001
    a = np.array([[1, t], [0, 1.0]])
    b = np.array([[0.5 * t * t], [t]])
    closed = a - b @ np.array([[kp, kd]])
    assert np.abs(np.linalg.eigvals(closed)).max() < 1.0
    # More position weight must give a stiffer loop.
    assert lqr_joint_gains(8.0e7, 2.0e3)[0] > kp


def test_computed_torque_tracks_a_fast_reference(sim):
    sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3)))
    ctl = ArmController(ArmModel(sim.model), sim.torque_limit)
    accels = np.zeros((20, ARM_DOF))
    accels[:10, 3], accels[10:, 3] = -30.0, 30.0
    accels[:10, 6], accels[10:, 6] = 40.0, -40.0
    traj = JointTrajectory(0.05, READY_POSTURE, np.zeros(ARM_DOF), accels, 0.01)
    worst = 0.0
    while sim.time < 0.45:
        q_ref, dq_ref, ddq_ref = traj.sample(sim.time)
        tau = ctl.computed_torque(sim.q, sim.dq, q_ref, dq_ref, ddq_ref)
        sim.step(tau)
        sim.step(tau)
        worst = max(worst, np.abs(q_ref - sim.q).max())
    assert worst < 0.01
    assert np.abs(traj.q[-1] - sim.q).max() < 0.002
    assert ctl.saturated_ticks == 0


def test_trajectory_is_continuous_and_integrates_its_accelerations():
    rng = np.random.default_rng(1)
    accels = rng.normal(0, 10, (6, ARM_DOF))
    traj = JointTrajectory(1.0, READY_POSTURE, np.zeros(ARM_DOF), accels, 0.02)
    for k in range(1, 6):
        before = traj.sample(1.0 + k * 0.02 - 1e-9)
        after = traj.sample(1.0 + k * 0.02 + 1e-9)
        assert before[0] == pytest.approx(after[0], abs=1e-6)
        assert before[1] == pytest.approx(after[1], abs=1e-6)
    q_end, dq_end, _ = traj.sample(10.0)
    assert q_end == pytest.approx(traj.q[-1])
    assert dq_end == pytest.approx(np.zeros(ARM_DOF))


PUSH_N = 12.0     # large against the 1 to 2 N that joint friction can hold at the cup


def test_impedance_is_soft_along_the_axis_and_stiff_across_it(sim):
    """Push the cup with the same force along and across its axis: it should give more along."""
    displacement = {}
    for name, direction_of in (("along", lambda ax: ax[:, 0]), ("across", lambda ax: ax[:, 1])):
        sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3)))
        arm = ArmModel(sim.model)
        # True joint state here: the test is about the impedance law. A Kalman velocity estimate
        # that does not model the push settles with a small bias and widens the deflection.
        cfg = ControllerConfig()
        cfg.sensing.observer = "exact"
        ctl = ArmController(arm, sim.torque_limit, cfg)
        x0, axis0 = sim.cup_position, sim.cup_axis
        arm.update(sim.q, sim.dq)
        push = PUSH_N * direction_of(arm.cup_axes)
        body = sim.model.body("catcher").id
        for _ in range(1500):
            tau = ctl.impedance(sim.q, sim.dq, x0, axis0, np.zeros(3), np.zeros(3), READY_POSTURE, np.zeros(ARM_DOF))
            sim.data.xfrc_applied[body, :3] = push
            sim.step(tau)
            sim.step(tau)
        displacement[name] = np.linalg.norm(sim.cup_position - x0)
        sim.data.xfrc_applied[body, :3] = 0.0
    gains = ImpedanceGains()
    # Static deflection is force over stiffness. Joint friction (0.3 N m) takes part of the load.
    assert displacement["along"] == pytest.approx(PUSH_N / gains.along_axis_n_per_m, rel=0.3)
    assert displacement["across"] == pytest.approx(PUSH_N / gains.lateral_n_per_m, rel=0.5)
    assert displacement["along"] > 4.0 * displacement["across"]


def test_prediction_matrices_match_step_by_step_integration():
    rng = np.random.default_rng(2)
    n, h = 9, 0.02
    sq, sv = prediction_matrices(n, h)
    u = rng.normal(0, 5, n)
    q, v = 0.3, -1.2
    for k in range(n):
        q, v = q + h * v + 0.5 * h * h * u[k], v + h * u[k]
        assert q == pytest.approx(0.3 + (k + 1) * h * -1.2 + sq[k] @ u)
        assert v == pytest.approx(-1.2 + sv[k] @ u)


def test_mpc_reaches_the_goal_and_respects_every_limit(sim):
    cfg = MpcConfig()
    mpc = ReachMpc(sim.joint_lo, sim.joint_hi, cfg)
    q_goal = READY_POSTURE + np.array([0.1, 0.05, 0.05, -0.5, -0.3, 0.4, 0.5, 0.3, -0.3, 0.2])
    dq_goal = np.array([0.0, 0.0, 0.0, 1.0, -0.5, 0.0, 1.5, 0.0, 0.5, 0.0])
    sol = mpc.solve(READY_POSTURE, np.zeros(ARM_DOF), q_goal, dq_goal, 0.35)
    assert sol.solved
    traj = JointTrajectory(0.0, READY_POSTURE, np.zeros(ARM_DOF), sol.accels, sol.step)
    assert traj.q[-1] == pytest.approx(q_goal, abs=3e-3)
    assert traj.dq[-1] == pytest.approx(dq_goal, abs=0.15)
    assert np.all(np.abs(sol.accels) <= cfg.accel_limit + 1e-6)
    assert np.all(np.abs(traj.dq) <= cfg.velocity_limit + 1e-3)
    assert np.all(traj.q >= np.minimum(mpc.q_lo, READY_POSTURE) - 1e-3)
    assert np.all(traj.q <= np.maximum(mpc.q_hi, READY_POSTURE) + 1e-3)


def test_mpc_stays_feasible_when_the_goal_is_out_of_reach_in_time(sim):
    mpc = ReachMpc(sim.joint_lo, sim.joint_hi)
    q_goal = READY_POSTURE.copy()
    q_goal[3] -= 1.5
    sol = mpc.solve(READY_POSTURE, np.zeros(ARM_DOF), q_goal, np.zeros(ARM_DOF), 0.05)
    assert sol.solved
    assert np.all(np.abs(sol.accels) <= mpc.cfg.accel_limit + 1e-6)
    assert abs(sol.terminal_position_error[3]) > 1.0      # reported, not hidden


def test_ik_puts_the_cup_on_target_and_aims_it(sim):
    ik = CatchIk(sim.model, sim.joint_lo, sim.joint_hi)
    rng = np.random.default_rng(3)
    dist = ThrowDistribution()
    position_errors, aim_errors = [], []
    for _ in range(60):
        point = rng.uniform(dist.aim_lo, dist.aim_hi)
        velocity = np.array([-5.0, rng.uniform(-1, 1), rng.uniform(-3, 1)])
        axis = -velocity / np.linalg.norm(velocity)
        q, err, _ = ik.solve(READY_POSTURE, point, axis, iterations=80)
        assert np.all(q >= ik.lo - 1e-9) and np.all(q <= ik.hi + 1e-9)
        position_errors.append(err)
        aim_errors.append(ik.axis_error_at_solution(axis))
    assert np.median(position_errors) < 1e-3
    assert np.percentile(position_errors, 90) < 5e-3
    assert np.median(aim_errors) < np.radians(3.0)


def test_velocity_goal_moves_the_cup_without_tilting_it(sim):
    ik = CatchIk(sim.model, sim.joint_lo, sim.joint_hi)
    axis = np.array([0.8, 0.0, 0.6])
    _, _, jac = ik.solve(READY_POSTURE, np.array([0.35, -0.2, 1.15]), axis, iterations=80)
    dq = ik.joint_velocity_for(jac, -1.2 * axis)
    assert jac[:3] @ dq == pytest.approx(-1.2 * axis, abs=0.02)
    assert jac[3:] @ dq == pytest.approx(np.zeros(2), abs=0.02)


def test_intercept_choice_is_reachable_and_refresh_keeps_its_time():
    cfg = InterceptConfig()
    times = 0.2 + 0.005 * np.arange(120)
    start = np.array([3.0, -0.2, 1.2])
    velocity = np.array([-5.0, 0.0, 2.6])      # crosses the workspace near its release height
    positions = start + np.outer(times, velocity) + np.outer(-0.5 * 9.81 * times**2, [0, 0, 1])
    velocities = velocity + np.outer(times, [0, 0, -9.81])
    covs = np.tile(1e-4 * np.eye(3), (len(times), 1, 1))
    hand = np.array([0.36, -0.14, 1.18])
    choice = choose_intercept(times, positions, velocities, covs, 0.2, hand, cfg)
    assert choice is not None
    reach = np.linalg.norm(choice.position - cfg.shoulder)
    assert cfg.reach_min_m < reach < cfg.reach_max_m
    assert choice.position[0] > cfg.min_forward_m
    moved = refresh_intercept(choice, choice.position + [0.0, 0.01, 0.0], choice.ball_velocity, 1e-4 * np.eye(3), 0.3, cfg)
    assert moved.time == choice.time
    assert refresh_intercept(choice, np.array([1.5, 0.0, 1.0]), choice.ball_velocity, 1e-4 * np.eye(3), 0.3, cfg) is None
    assert choose_intercept(times, positions + [0, 2.0, 0], velocities, covs, 0.2, hand, cfg) is None


def test_ilqr_matches_lqr_and_a_bounded_least_squares_solution():
    t, n = 0.02, 40
    a = np.array([[1, t], [0, 1.0]])
    b = np.array([[0.5 * t * t], [t]])
    qf, r = np.diag([1e4, 1e2]), np.array([1e-2])
    x_goal = np.array([1.0, 0.5])
    phi = np.column_stack([np.linalg.matrix_power(a, n - 1 - k) @ b[:, 0] for k in range(n)])
    stacked_a = np.vstack([np.sqrt(qf) @ phi, np.sqrt(r[0]) * np.eye(n)])
    stacked_b = np.concatenate([np.sqrt(qf) @ x_goal, np.zeros(n)])
    step, lin = (lambda x, u: a @ x + b @ u), (lambda x, u: (a, b))
    for bound in (1e9, 3.0):
        reference = lsq_linear(stacked_a, stacked_b, bounds=(-bound, bound), tol=1e-13).x
        res = ilqr.solve(step, lin, np.zeros(2), np.zeros((n, 1)), x_goal, r, qf, np.array([-bound]), np.array([bound]),
                         ilqr.IlqrOptions(cost_tolerance=1e-12, max_iterations=200))
        assert res.controls[:, 0] == pytest.approx(reference, abs=1e-6)
        assert np.all(np.diff(res.cost_history) <= 1e-12)


def test_rls_recovers_a_mass_and_follows_a_change():
    rng = np.random.default_rng(4)
    est = PayloadEstimator(forgetting=0.99)
    estimate = 0.0
    for mass in (0.3, 0.8):
        for _ in range(800):
            accel = rng.normal(0, 0.5, 3)
            phi = accel - np.array([0, 0, -9.81])
            force = (0.09 + mass) * phi + rng.normal(0, 0.05, 3)
            estimate = est.update(force, accel)
        assert estimate == pytest.approx(mass, abs=0.005)


def test_rls_variance_stays_bounded_without_excitation():
    rls = ScalarRls(forgetting=0.9, max_variance=10.0)
    for _ in range(1000):
        rls.update(np.zeros(3), np.zeros(3))
    assert rls.variance <= 10.0
    assert rls.theta == 0.0
