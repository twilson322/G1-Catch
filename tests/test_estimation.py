import numpy as np
import pytest

from g1catch import ballistics as bal
from g1catch import vision as vis
from g1catch.imm import BallTracker, TrackerConfig
from g1catch.sim import CatchSim, ThrowDistribution
from g1catch.ukf import Measurement, propagate, sigma_points, white_acceleration_noise

ARM_AWAY = np.array([0.0, 0.0, 0.0, 0.5, -1.5, 0.0, -0.3, 0.0, 0.0, 0.0])
FRAME_TICKS = 33
PIXEL_SIGMA = 0.4


def test_sigma_points_reproduce_mean_and_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T + np.eye(6)
    mean = rng.normal(size=6)
    pts, w = sigma_points(mean, cov)
    assert w.sum() == pytest.approx(1.0)
    assert w @ pts == pytest.approx(mean)
    dev = pts - mean
    assert (dev.T * w) @ dev == pytest.approx(cov)


def test_process_noise_matches_the_white_acceleration_integral():
    q = white_acceleration_noise(2.0, 0.1)
    assert q[0, 0] == pytest.approx(2.0 * 0.1**3 / 3)
    assert q[0, 3] == pytest.approx(2.0 * 0.1**2 / 2)
    assert q[3, 3] == pytest.approx(2.0 * 0.1)
    assert np.all(np.linalg.eigvalsh(q) > -1e-15)


def test_propagation_does_not_depend_on_how_the_interval_is_split():
    state = np.array([[3.0, 0.0, 1.2, -5.0, 0.3, 2.0]])
    whole = propagate(state, 0.2, bal.NOMINAL_KD)
    halves = propagate(propagate(state, 0.1, bal.NOMINAL_KD), 0.1, bal.NOMINAL_KD)
    assert whole == pytest.approx(halves, abs=1e-9)


def _track(throw, sim, intr, rng, cfg, log_from=5):
    sim.reset(throw, ARM_AWAY)
    poses = vis.camera_poses(sim.model, sim.data)
    tracker = BallTracker(intr, cfg)
    nees, nis, final_error = [], [], np.nan
    end = np.floor(throw.flight_time / (FRAME_TICKS * sim.dt)) * FRAME_TICKS * sim.dt
    while sim.time < end - 1e-9:
        pixels = []
        for pose in poses:
            px, depth = vis.project(intr, pose, sim.ball_position)
            pixels.append(px + rng.normal(0, PIXEL_SIGMA, 2) if depth > 0.15 and vis.in_image(intr, px) else None)
        meas = Measurement(tuple(pixels), poses)
        if not tracker.initialised:
            tracker.initialise(meas, sim.time)
        elif tracker.step(meas, sim.time) and tracker.updates > log_from:
            err = tracker.mean - np.concatenate([sim.ball_position, sim.ball_velocity])
            nees.append(err @ np.linalg.solve(tracker.covariance, err))
            nis.append(tracker.last_nis)
            final_error = np.linalg.norm(err[:3])
        for _ in range(FRAME_TICKS):
            sim.step(np.zeros(10))
    return nees, nis, final_error


@pytest.fixture(scope="module")
def world():
    sim = CatchSim()
    return sim, vis.Intrinsics.from_fovy(sim.model.cam_fovy[0])


def test_tracker_is_accurate_and_consistent_on_direct_throws(world):
    sim, intr = world
    rng = np.random.default_rng(11)
    dist = ThrowDistribution()
    nees, nis, finals = [], [], []
    for _ in range(25):
        a, b, c = _track(dist.sample(rng), sim, intr, rng, TrackerConfig(pixel_sigma=PIXEL_SIGMA))
        nees += a
        nis += b
        finals.append(c)
    assert np.median(finals) < 0.01            # 1 cm when the ball arrives
    # A consistent filter averages 6 (state) and 4 (measurement). Allow conservative, not overconfident.
    assert 2.0 < np.mean(nees) < 9.0
    assert 2.5 < np.mean(nis) < 5.5


def test_multiple_model_filter_stays_consistent_through_a_bounce(world):
    sim, intr = world
    dist = ThrowDistribution(bounce=True)
    means = {}
    for use_imm in (True, False):
        rng = np.random.default_rng(12)
        nees = []
        for _ in range(25):
            a, _, _ = _track(dist.sample(rng), sim, intr, rng, TrackerConfig(pixel_sigma=PIXEL_SIGMA, use_imm=use_imm))
            nees += a
        means[use_imm] = np.mean(nees)
    assert means[True] < 9.0
    assert means[False] > means[True]          # the single low-noise filter is overconfident at impact


def test_gate_rejects_a_wild_measurement(world):
    sim, intr = world
    rng = np.random.default_rng(5)
    throw = ThrowDistribution().sample(rng)
    sim.reset(throw, ARM_AWAY)
    poses = vis.camera_poses(sim.model, sim.data)
    tracker = BallTracker(intr, TrackerConfig(pixel_sigma=PIXEL_SIGMA))
    for k in range(12):
        pixels = tuple(vis.project(intr, pose, sim.ball_position)[0] + rng.normal(0, PIXEL_SIGMA, 2) for pose in poses)
        if not tracker.initialised:
            tracker.initialise(Measurement(pixels, poses), sim.time)
        else:
            assert tracker.step(Measurement(pixels, poses), sim.time)
        for _ in range(FRAME_TICKS):
            sim.step(np.zeros(10))
    before = tracker.mean.copy()
    bogus = tuple(vis.project(intr, pose, sim.ball_position)[0] + np.array([60.0, -45.0]) for pose in poses)
    assert not tracker.step(Measurement(bogus, poses), sim.time)
    assert np.linalg.norm(tracker.mean[:3] - sim.ball_position) < np.linalg.norm(before[:3] - sim.ball_position) + 0.2


def test_forecast_with_lead_time_equals_a_later_forecast_from_the_same_estimate(world):
    sim, intr = world
    rng = np.random.default_rng(6)
    throw = ThrowDistribution().sample(rng)
    sim.reset(throw, ARM_AWAY)
    poses = vis.camera_poses(sim.model, sim.data)
    tracker = BallTracker(intr, TrackerConfig(pixel_sigma=PIXEL_SIGMA))
    for _ in range(8):
        pixels = tuple(vis.project(intr, pose, sim.ball_position)[0] for pose in poses)
        meas = Measurement(pixels, poses)
        tracker.step(meas, sim.time) if tracker.initialised else tracker.initialise(meas, sim.time)
        for _ in range(FRAME_TICKS):
            sim.step(np.zeros(10))
    t = tracker.time
    _, direct, _, _ = tracker.forecast(t, 0.2, 1)
    _, led, _, _ = tracker.forecast(t + 0.03, 0.17, 1)
    assert led[0] == pytest.approx(direct[0], abs=2e-3)


def test_symmetric_positive_repairs_an_indefinite_covariance():
    from g1catch.ukf import MIN_COVARIANCE_EIGENVALUE, symmetric_positive
    rng = np.random.default_rng(8)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T
    values, vectors = np.linalg.eigh(cov)
    values[0] = -1e-6                                   # what round-off can leave after an update
    broken = (vectors * values) @ vectors.T
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(broken)
    fixed = symmetric_positive(broken)
    np.linalg.cholesky(fixed)
    assert np.linalg.eigvalsh(fixed).min() >= 0.5 * MIN_COVARIANCE_EIGENVALUE
    healthy = cov + np.eye(6)
    assert symmetric_positive(healthy) == pytest.approx(healthy)


def test_coasting_in_open_air_does_not_assume_an_impact(world):
    """With no measurements for a quarter second, uncertainty must grow like free flight does,
    not like a ball that may have bounced off something."""
    sim, intr = world
    rng = np.random.default_rng(9)
    throw = ThrowDistribution().sample(rng)
    growth = {}
    for name, cfg in (("height-aware", TrackerConfig(pixel_sigma=PIXEL_SIGMA)),
                      ("one matrix", TrackerConfig(pixel_sigma=PIXEL_SIGMA,
                                                   mode_transition_open_air=np.array([[0.90, 0.10], [0.60, 0.40]])))):
        sim.reset(throw, ARM_AWAY)
        poses = vis.camera_poses(sim.model, sim.data)
        tracker = BallTracker(intr, cfg)
        meas_rng = np.random.default_rng(10)
        for _ in range(14):
            pixels = tuple(vis.project(intr, pose, sim.ball_position)[0] + meas_rng.normal(0, PIXEL_SIGMA, 2) for pose in poses)
            meas = Measurement(pixels, poses)
            tracker.step(meas, sim.time) if tracker.initialised else tracker.initialise(meas, sim.time)
            for _ in range(FRAME_TICKS):
                sim.step(np.zeros(10))
        before = np.sqrt(np.trace(tracker.covariance[:3, :3]))
        for _ in range(15):                               # out of view
            tracker.step(None, sim.time)
            for _ in range(FRAME_TICKS):
                sim.step(np.zeros(10))
        growth[name] = np.sqrt(np.trace(tracker.covariance[:3, :3])) / before
        assert tracker.mean[2] > 0.5                      # still in open air, so the comparison is fair
        error = np.linalg.norm(tracker.mean[:3] - sim.ball_position)
        assert error < 3.0 * np.sqrt(np.trace(tracker.covariance[:3, :3]))
    assert growth["one matrix"] > 2.0 * growth["height-aware"]
