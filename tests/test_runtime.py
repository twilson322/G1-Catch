import numpy as np
import pytest

from g1catch.experiment import Trial, catch_summary, run_trial, wilson_interval
from g1catch.runtime import RuntimeConfig
from g1catch.sim import CatchSim, Throw, ThrowDistribution


def test_same_seed_gives_the_same_throws_and_outcomes():
    a = run_trial(Trial("a", RuntimeConfig(), ThrowDistribution(), 3, seed=42))
    b = run_trial(Trial("b", RuntimeConfig(), ThrowDistribution(), 3, seed=42))
    for ra, rb in zip(a, b):
        assert ra["caught"] == rb["caught"]
        assert ra["miss_cm"] == pytest.approx(rb["miss_cm"], abs=1e-9)
        assert ra["flight_time"] == rb["flight_time"]


def test_pipeline_catches_most_direct_throws():
    """End to end with the pixel-noise sensor. A regression bound, well below the measured rate."""
    rows = run_trial(Trial("regression", RuntimeConfig(), ThrowDistribution(), 30, seed=7))
    summary = catch_summary(rows)
    assert summary["rate"] >= 0.6
    assert summary["median_miss_cm"] < 4.0


def test_without_velocity_matching_the_ball_bounces_out():
    cfg = RuntimeConfig()
    cfg.intercept.velocity_match = 0.0
    rows = run_trial(Trial("stationary cup", cfg, ThrowDistribution(), 20, seed=7))
    assert catch_summary(rows)["rate"] <= 0.2
    assert np.mean([r["touched"] for r in rows]) > 0.8     # it gets there, it just cannot keep the ball


def test_throw_out_of_reach_is_not_committed_to():
    dist = ThrowDistribution(aim_lo=(0.3, 1.2, 1.0), aim_hi=(0.4, 1.4, 1.2))     # a metre to the left
    rows = run_trial(Trial("far left", RuntimeConfig(), dist, 4, seed=1))
    assert all(r["committed"] == 0 and r["caught"] == 0 for r in rows)


def test_caught_requires_the_ball_inside_the_cup_volume():
    sim = CatchSim()
    sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3)))
    assert not sim.ball_in_cup()
    sim.data.qpos[10:13] = sim.cup_position
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.ball_in_cup()
    sim.data.qpos[10:13] = sim.cup_position + 0.09 * sim.cup_axis     # above the rim
    mujoco.mj_forward(sim.model, sim.data)
    assert not sim.ball_in_cup()


def test_wilson_interval_brackets_the_proportion():
    lo, hi = wilson_interval(85, 100)
    assert lo < 0.85 < hi
    assert 0.76 < lo < 0.78 and 0.90 < hi < 0.92
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12)


def test_catch_time_is_not_reoptimised_every_frame():
    """Choosing a new catch time on every frame lets it creep along the ball's path. The time may
    change when a new forecast puts the ball out of reach at the old one, and only then."""
    rows = run_trial(Trial("frames", RuntimeConfig(), ThrowDistribution(), 10, seed=3, keep_frames=True))
    change_counts = []
    for row in rows:
        planned = [f["plan_t"] for f in row["_frames"] if not np.isnan(f["plan_t"])]
        if len(planned) < 6:
            continue
        changes = int(np.sum(np.abs(np.diff(planned)) > 1e-9))
        change_counts.append(changes)
        assert changes <= len(planned) // 3
    assert len(change_counts) >= 5
    assert np.median(change_counts) <= 2


def test_heavier_ball_rests_in_the_cup_without_exciting_the_contact():
    """Changing the ball's mass needs MuJoCo's derived constants recomputed, or the contact goes unstable."""
    import mujoco

    from g1catch.control import ArmController, ArmModel
    from g1catch.intercept import CatchIk
    from g1catch.sim import ARM_DOF, READY_POSTURE
    sim = CatchSim()
    ik = CatchIk(sim.model, sim.joint_lo, sim.joint_hi)
    q_hold, _, _ = ik.solve(READY_POSTURE, np.array([0.35, -0.20, 1.10]), np.array([0.0, 0.0, 1.0]), iterations=100)
    sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3), mass=0.3), q_hold)
    sim.data.qpos[ARM_DOF:ARM_DOF + 3] = sim.cup_position - 0.001 * sim.cup_axis
    mujoco.mj_forward(sim.model, sim.data)
    ctl = ArmController(ArmModel(sim.model), sim.torque_limit)
    for _ in range(500):
        tau = ctl.computed_torque(sim.q, sim.dq, q_hold, np.zeros(ARM_DOF), np.zeros(ARM_DOF))
        sim.step(tau)
        sim.step(tau)
    assert sim.ball_in_cup()
    assert np.linalg.norm(sim.ball_velocity) < 0.05


def test_catch_axis_never_opens_below_the_minimum_tilt():
    from g1catch.intercept import catch_axis
    falling = np.array([-5.0, 0.3, -2.0])
    assert catch_axis(falling, 0.35) == pytest.approx(-falling / np.linalg.norm(falling))
    rising = np.array([-5.0, 0.5, 2.0])
    axis = catch_axis(rising, 0.35)
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert axis[2] == pytest.approx(0.35)
    assert axis[:2] @ -rising[:2] > 0                      # still faces the ball horizontally
    assert catch_axis(np.array([0.0, 0.0, 3.0]), 0.35) == pytest.approx([0.0, 0.0, 1.0])


def test_wrong_mass_model_changes_dynamics_and_keeps_kinematics():
    from g1catch.control import ArmModel
    from g1catch.sim import ARM_DOF, READY_POSTURE, model_with_mass_error
    sim = CatchSim()
    wrong = model_with_mass_error(sim.model, 0.3, np.random.default_rng(1))
    true_arm, wrong_arm = ArmModel(sim.model), ArmModel(wrong)
    for arm in (true_arm, wrong_arm):
        arm.update(READY_POSTURE, np.zeros(ARM_DOF))
    assert wrong_arm.cup_position == pytest.approx(true_arm.cup_position)
    ratio = np.diag(wrong_arm.mass_matrix) / np.diag(true_arm.mass_matrix)
    assert np.abs(ratio - 1.0).max() > 0.03 and np.all((ratio > 0.69) & (ratio < 1.31))
    assert sim.model.body_mass == pytest.approx(CatchSim().model.body_mass)     # the true model is untouched
    same = model_with_mass_error(sim.model, 0.0, np.random.default_rng(1))
    assert same.body_mass == pytest.approx(sim.model.body_mass)


def test_planner_refuses_a_configuration_inside_the_body():
    from g1catch.runtime import CatchRunner
    from g1catch.sensing import SyntheticSensor
    from g1catch.sim import READY_POSTURE
    from g1catch.vision import Intrinsics
    sim = CatchSim()
    runner = CatchRunner(sim, SyntheticSensor(Intrinsics.from_fovy(sim.model.cam_fovy[0]), np.random.default_rng(0)))
    assert not runner._collides(READY_POSTURE)
    folded = READY_POSTURE.copy()
    folded[4] = 0.6                                         # shoulder rolled into the torso
    assert runner._collides(folded)


def test_track_starts_on_the_ball_when_a_brighter_decoy_is_listed_first():
    """Detections come strongest first. A decoy on the far wall must not be picked."""
    from g1catch import vision as vis
    from g1catch.imm import BallTracker
    from g1catch.ukf import Measurement
    sim = CatchSim()
    sim.reset(Throw(np.array([3.2, 0.2, 1.4]), np.zeros(3)))
    intr = vis.Intrinsics.from_fovy(sim.model.cam_fovy[0])
    poses = vis.camera_poses(sim.model, sim.data)
    ball = [vis.project(intr, pose, sim.ball_position)[0] for pose in poses]
    decoy = [vis.project(intr, pose, np.array([6.3, 1.4, 1.5]))[0] for pose in poses]
    meas = Measurement((decoy[0], decoy[1]), poses, candidates=([decoy[0], ball[0]], [decoy[1], ball[1]]))
    tracker = BallTracker(intr)
    assert tracker.initialise(meas, 0.0)
    assert tracker.mean[:3] == pytest.approx(sim.ball_position, abs=1e-6)
    only_decoy = Measurement((decoy[0], decoy[1]), poses, candidates=([decoy[0]], [decoy[1]]))
    assert not BallTracker(intr).initialise(only_decoy, 0.0)


def test_vergence_error_biases_the_forecast_in_range():
    cfg = RuntimeConfig()
    cfg.stereo_vergence_error_deg = 0.4
    clean = run_trial(Trial("clean", RuntimeConfig(), ThrowDistribution(), 8, seed=9))
    biased = run_trial(Trial("biased", cfg, ThrowDistribution(), 8, seed=9))
    median = lambda rows: np.nanmedian([r["forecast_error_cm"] for r in rows])
    assert median(biased) > 2.0 * median(clean)


@pytest.mark.render
def test_rendered_pipeline_with_the_cnn_catches_most_nominal_throws():
    """End to end from rendered images through the trained detector. Pins the detector weights,
    the renderer and the multi-candidate track start together."""
    pytest.importorskip("torch")
    try:
        rows = run_trial(Trial("rendered", RuntimeConfig(), ThrowDistribution(), 10, seed=500, sensor="cnn"))
    except Exception as exc:                      # no GL context on this machine
        if "GL" in type(exc).__name__ or "gl" in str(exc).lower():
            pytest.skip(f"offscreen rendering unavailable: {exc}")
        raise
    assert sum(r["caught"] for r in rows) >= 8
