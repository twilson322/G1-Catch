"""The compiled flight model must be the NumPy reference, to round-off."""
import numpy as np
import pytest

from g1catch import ballistics as bal
from g1catch import ukf

core = pytest.importorskip("g1catch._flight_core")


def random_states(rng, n=40):
    states = np.empty((n, 6))
    states[:, :3] = rng.uniform([0.3, -1.0, 0.04], [4.0, 1.0, 2.2], (n, 3))
    states[:, 3:] = rng.uniform([-8.0, -2.0, -7.0], [0.0, 2.0, 4.0], (n, 3))
    return states


@pytest.mark.parametrize("dt", [0.0005, 0.004, 0.0165, 0.33])
@pytest.mark.parametrize("kd", [0.0, bal.NOMINAL_KD])
def test_propagate_matches_reference(dt, kd):
    states = random_states(np.random.default_rng(0))
    assert ukf.propagate(states, dt, kd) == pytest.approx(ukf.propagate_reference(states, dt, kd), abs=1e-12)


def test_rollout_matches_reference_through_bounces():
    states = random_states(np.random.default_rng(1))
    compiled = ukf.rollout(states, 0.005, 200, bal.NOMINAL_KD)
    reference = np.empty_like(compiled)
    current = states
    for k in range(200):
        current = ukf.propagate_reference(current, 0.005, bal.NOMINAL_KD)
        reference[k] = current
    assert compiled == pytest.approx(reference, abs=1e-10)
    assert (reference[:, :, 5] > 0).any() and reference[:, :, 2].min() < 0.1     # some of them did bounce


def test_inputs_are_not_modified_and_bad_shapes_are_refused():
    states = random_states(np.random.default_rng(2))
    before = states.copy()
    ukf.propagate(states, 0.02, bal.NOMINAL_KD)
    assert np.array_equal(states, before)
    with pytest.raises(ValueError):
        core.propagate(np.zeros((4, 5)), 0.01, ukf.flight_params(0.0), 0.004)
    with pytest.raises(ValueError):
        core.rollout(np.zeros((4, 6)), 0.01, 0, ukf.flight_params(0.0), 0.004)
