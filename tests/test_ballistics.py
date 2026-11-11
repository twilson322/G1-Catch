import numpy as np
import pytest

from g1catch import ballistics as bal
from g1catch.sim import CatchSim, Throw, ThrowDistribution

ARM_AWAY = np.array([0.0, 0.0, 0.0, 0.5, -1.5, 0.0, -0.3, 0.0, 0.0, 0.0])


def test_drag_free_flight_is_a_parabola():
    p0, v0 = np.array([3.0, 0.0, 1.0]), np.array([-5.0, 0.5, 3.0])
    ps, vs = bal.rollout(p0, v0, 0.002, 200, kd=0.0, floor=False)
    t = 0.4
    assert ps[-1] == pytest.approx(p0 + v0 * t + 0.5 * np.array([0, 0, -bal.GRAVITY_M_S2]) * t**2, abs=1e-9)
    assert vs[-1] == pytest.approx(v0 + np.array([0, 0, -bal.GRAVITY_M_S2]) * t, abs=1e-9)


def test_drag_removes_energy_and_never_adds_it():
    p, v = np.array([0.0, 0.0, 50.0]), np.array([8.0, 0.0, 0.0])
    energy = lambda p, v: 0.5 * v @ v + bal.GRAVITY_M_S2 * p[2]
    previous = energy(p, v)
    for _ in range(300):
        p, v = bal.step(p, v, 0.004, floor=False)
        assert energy(p, v) < previous
        previous = energy(p, v)


def test_terminal_velocity_matches_the_closed_form():
    p, v = np.array([0.0, 0.0, 1e4]), np.zeros(3)
    for _ in range(20000):
        p, v = bal.step(p, v, 0.004, floor=False)
    assert -v[2] == pytest.approx(np.sqrt(bal.GRAVITY_M_S2 / bal.NOMINAL_KD), rel=1e-6)


def test_simulated_flight_matches_the_model_when_drag_is_known():
    """The simulator's aerodynamic force and the filter's model are the same physics."""
    sim = CatchSim()
    throw = ThrowDistribution().sample(np.random.default_rng(3))
    throw.spin[:] = 0.0
    throw.drag_coefficient = bal.NOMINAL_DRAG_COEFFICIENT
    sim.reset(throw, ARM_AWAY)
    for _ in range(int(0.4 / sim.dt)):
        sim.step(np.zeros(10))
    ps, _ = bal.rollout(throw.position, throw.velocity, 0.001, 400)
    assert np.linalg.norm(ps[-1] - sim.ball_position) < 0.003
    drag_free, _ = bal.rollout(throw.position, throw.velocity, 0.001, 400, kd=0.0)
    assert np.linalg.norm(drag_free[-1] - sim.ball_position) > 0.015     # drag is not negligible


def test_nominal_bounce_parameters_match_the_simulator():
    sim = CatchSim()
    ratios = []
    for vz, vx in ((-4.0, 2.0), (-6.0, 3.0), (-3.0, 4.0)):
        sim.reset(Throw(np.array([2.0, 1.5, 0.3]), np.array([vx, 0.0, vz]), drag_coefficient=0.0), ARM_AWAY)
        v_in = None
        for _ in range(800):
            if v_in is None and sim.ball_position[2] < bal.BALL_RADIUS_M + 1e-3:
                v_in = sim.ball_velocity
            sim.step(np.zeros(10))
            if v_in is not None and sim.ball_position[2] > 0.06 and sim.ball_velocity[2] > 0:
                break
        v_out = sim.ball_velocity
        ratios.append((-v_out[2] / v_in[2], v_out[0] / v_in[0]))
    restitution, retention = np.mean(ratios, axis=0)
    assert restitution == pytest.approx(bal.NOMINAL_RESTITUTION, abs=0.02)
    assert retention == pytest.approx(bal.NOMINAL_TANGENTIAL_RETENTION, abs=0.03)


def test_bounce_throws_hit_the_floor_once_and_reach_the_aim_height():
    rng = np.random.default_rng(1)
    dist = ThrowDistribution(bounce=True)
    for _ in range(20):
        throw = dist.sample(rng)
        steps = int(round(throw.flight_time / 0.002))
        ps, _ = bal.rollout(throw.position, throw.velocity, 0.002, steps, kd=0.0)
        assert ps[:, 2].min() < 0.06
        assert np.linalg.norm(ps[-1] - throw.aim) < 0.06
