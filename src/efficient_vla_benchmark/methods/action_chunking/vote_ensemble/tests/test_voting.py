import numpy as np
import pytest

from ..codec import decode, encode
from ..config import VoteConfig
from ..implementations.cogact.api import validate_experiment
from ..method import VoteEnsemble, ensemble_vote


def test_majority_tie_and_strict_threshold_equations():
    result, count = ensemble_vote([[1, 0], [2, 0], [-1, 0]], 0.5)
    np.testing.assert_array_equal(result, [1.5, 0])
    assert count == 2
    # Tie selects NOT-similar, not the current prediction.
    result, count = ensemble_vote([[1, 0], [-1, 0]], 0.5)
    np.testing.assert_array_equal(result, [1, 0])
    assert count == 1
    result, _ = ensemble_vote([[1, 0], [0, 1]], 0)
    np.testing.assert_array_equal(result, [1, 0])


def test_zero_vectors_and_singleton_are_finite():
    for candidates in ([[0, 0]], [[0, 0], [0, 0]], [[0, 0], [1, 0]]):
        result, count = ensemble_vote(candidates, 0.5)
        assert np.isfinite(result).all() and count >= 1


def test_alignment_uses_future_predictions_not_executed_history():
    method = VoteEnsemble(VoteConfig(history_chunks=2), lambda a: a, lambda a: a)
    method.on_plan(np.array([[9.0, 9.0], [1, 0], [2, 0]]), 0)
    method.on_plan(np.array([[-1.0, 0], [3, 0], [4, 0]]), 1)
    np.testing.assert_array_equal(method.transform_action(np.zeros(2), 1), [1, 0])
    assert method.candidate_count == 2
    # Both old predictions agree for physical t=2.
    np.testing.assert_array_equal(method.transform_action(np.zeros(2), 2), [2.5, 0])
    # Expire the first chunk; t=3 has only the newer chunk's third prediction.
    np.testing.assert_array_equal(method.transform_action(np.zeros(2), 3), [4, 0])
    method.reset(None)
    assert not method.predictions
    with pytest.raises(RuntimeError, match="aligned prediction"):
        method.transform_action(np.zeros(2), 0)


def test_history_bound_and_disabled_identity():
    method = VoteEnsemble(VoteConfig.parse({"history_chunks": 2}), lambda a: a, lambda a: a)
    for t in range(50):
        method.on_plan(np.ones((8, 7)), t)
        method.transform_action(np.ones(7), t)
        assert len(method.predictions) <= 3
    disabled = VoteEnsemble(VoteConfig.parse({"enabled": False}), encode, decode)
    chunk = np.ones((8, 7))
    disabled.on_plan(chunk, 0)
    np.testing.assert_array_equal(disabled.transform_action(chunk[0], 0), chunk[0])
    assert not disabled.predictions


def test_oft_codec_roundtrip_and_majority_gripper():
    actions = np.array([[0.2, 0, 0, 0, 0, 0, -1], [0.1, 0, 0, 0, 0, 0, 1.0]])
    original = actions.copy()
    encoded = encode(actions)
    np.testing.assert_array_equal(encoded[:, -1], [1, 0])
    for before, after in zip(actions, encoded):
        np.testing.assert_array_equal(decode(after), before)
    np.testing.assert_array_equal(actions, original)
    assert decode(np.array([0, 0, 0, 0, 0, 0, 0.7]))[-1] == -1


def test_cadence_and_history_rejected_before_model_loading():
    from efficient_vla_benchmark.core.config import ExperimentConfig

    with pytest.raises(ValueError, match="horizon=8"):
        validate_experiment(ExperimentConfig("cogact", execution_horizon=1), VoteConfig())
    with pytest.raises(ValueError, match="one overlapping"):
        validate_experiment(
            ExperimentConfig("cogact", execution_horizon=8), VoteConfig(history_chunks=2)
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"history_chunks": 0},
        {"history_chunks": True},
        {"max_history": 3},
        {"similarity_threshold": 1},
        {"similarity_threshold": float("nan")},
        {"enabled": 1},
        {"unknown": 1},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        VoteConfig.parse(parameters)
