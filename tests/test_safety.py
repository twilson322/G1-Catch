import numpy as np
import pytest

from g1catch.joints import JointObserver, JointSensingConfig, kalman_gain, observer_model, quantize
from g1catch.safety import filter_acceleration

LO, HI = np.array([-1.0]), np.array([1.0])
A_MAX, V_MAX, STEP = np.array([50.0]), np.array([8.0]), 0.0165


def test_filter_leaves_a_safe_command_alone():
    u, changed = filter_acceleration(np.array([0.0]), np.array([0.5]), np.array([10.0]), STEP, LO, HI, A_MAX, V_MAX)
    assert u == pytest.approx([10.0])
    assert not changed[0]


def test_filtered_joint_never_leaves_its_range_under_an_adversarial_command():
    """Whatever is asked for, a joint that starts inside its range stays inside."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(-0.9, 0.9, 1)
        dq = rng.uniform(-3.0, 3.0, 1)
        # Start from a state that can still stop, which is the set the barrier keeps invariant.
        if q[0] + max(dq[0], 0) ** 2 / (2 * 45.0) > 1.0 or q[0] - max(-dq[0], 0) ** 2 / (2 * 45.0) < -1.0:
            continue
        push = rng.choice([-1.0, 1.0]) * A_MAX          # full acceleration toward one limit, forever
        for _ in range(120):
            u, _ = filter_acceleration(q, dq, push, STEP, LO, HI, A_MAX, V_MAX)
            assert abs(u[0]) <= A_MAX[0] + 1e-9
            q = q + STEP * dq + 0.5 * STEP**2 * u
            dq = dq + STEP * u
            assert LO[0] - 1e-6 <= q[0] <= HI[0] + 1e-6
            assert abs(dq[0]) <= V_MAX[0] + 1e-6


def test_unfiltered_command_does_leave_the_range():
    q, dq = np.array([0.5]), np.array([0.0])
    for _ in range(40):
        q = q + STEP * dq + 0.5 * STEP**2 * A_MAX
        dq = dq + STEP * A_MAX
    assert q[0] > HI[0]


def test_quantizer_rounds_to_encoder_counts():
    step = 2 * np.pi / 2**15
    q = np.array([0.1234567, -1.0])
    assert np.abs(quantize(q, 15) - q).max() <= 0.5 * step + 1e-15
    assert quantize(q, 15) / step == pytest.approx(np.round(quantize(q, 15) / step))


@pytest.mark.parametrize("kind", ["kalman", "eso"])
def test_observer_error_dynamics_are_stable(kind):
    with_d = kind == "eso"
    a, _, c = observer_model(0.001, with_d)
    gain = kalman_gain(0.001, JointSensingConfig(), with_d)
    poles = np.linalg.eigvals(a - np.outer(a @ gain, c[0]))
    assert np.abs(poles).max() < 1.0


@pytest.mark.parametrize("kind", ["difference", "kalman", "eso"])
def test_observers_recover_velocity_from_a_quantized_angle(kind):
    """A joint following a sinusoid, seen through a 15-bit encoder."""
    obs = JointObserver(1, 0.001, JointSensingConfig(observer=kind))
    w, amp = 2 * np.pi * 2.0, 0.5
    obs.reset(np.array([0.0]))
    errors = []
    for k in range(2000):
        t = k * 0.001
        q_true = np.array([amp * np.sin(w * t)])
        dq_true = np.array([amp * w * np.cos(w * t)])
        accel = np.array([-amp * w * w * np.sin(w * t)])
        _, dq_hat = obs.correct(q_true, dq_true)
        obs.predict(accel)
        if k > 300:
            errors.append(abs(dq_hat[0] - dq_true[0]))
    # Peak speed is 6.3 rad/s. The model-based observers know the acceleration and do better.
    assert np.max(errors) < (0.25 if kind == "difference" else 0.06)


def test_extended_state_observer_finds_a_constant_disturbance():
    """The estimate rides on quantization noise, so its mean is compared and its spread is bounded."""
    obs = JointObserver(1, 0.001, JointSensingConfig(observer="eso"))
    q, dq, disturbance = np.array([0.2]), np.array([0.0]), 4.0       # rad/s^2 the model does not know about
    obs.reset(q)
    estimates = []
    for _ in range(1000):
        obs.correct(q, dq)
        command = np.array([-3.0 * q[0] - 2.0 * dq[0]])
        obs.predict(command)
        q = q + 0.001 * dq + 0.5e-6 * (command + disturbance)
        dq = dq + 0.001 * (command + disturbance)
        estimates.append(obs.disturbance[0])
    assert estimates[60] > 0.6 * disturbance                         # responds within 60 ms
    assert np.mean(estimates[-400:]) == pytest.approx(disturbance, rel=0.03)
    assert np.std(estimates[-400:]) < 1.0
