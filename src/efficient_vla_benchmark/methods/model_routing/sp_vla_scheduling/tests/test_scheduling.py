import numpy as np
import pytest

from ..config import SPVLAConfig
from ..method import SPVLAScheduling, extrapolate


def test_ridge_matches_independent_reference_and_holds_rotation_gripper():
    history = np.array([[0.1 * i, 0.02 * i, 0, 0.3, 0.4, 0.5, -1] for i in range(6)])
    x = np.vstack([np.arange(6), np.ones(6)]).T
    beta = np.linalg.solve(x.T @ x + 1e-4 * np.eye(2), x.T @ history[:, :3])
    expected = np.clip(beta[0] * 6 + beta[1], -0.9, 0.9)
    delta = np.max(np.abs(np.diff(history[:, :3], axis=0)), axis=0)
    expected = history[-1, :3] + np.clip(expected - history[-1, :3], -delta, delta)
    result = extrapolate(history, 1e-4, 0.9)
    np.testing.assert_array_equal(result[:3], expected)
    np.testing.assert_array_equal(result[3:], history[-1, 3:])


def test_history_counts_executed_not_predicted_actions_and_forces_refresh():
    method = SPVLAScheduling(SPVLAConfig.parse({}))
    action = np.array([0.2, 0.1, 0.01, 0, 0, 0, 1.0])
    method.on_plan(np.tile(action, (8, 1)), 0)
    assert len(method.history) == 0
    for step in range(8):
        method.after_action(action, 7 - step)
    fits = 0
    while (fitted := method.before_plan(None, 8 + fits)) is not None:
        np.testing.assert_allclose(fitted, action, atol=1e-12)
        method.after_action(fitted, 0)
        fits += 1
        assert fits <= 3
    assert fits == 3
    assert sum(method.origins) == 3
    method.reset(None)
    assert not method.history and method.replacements == 0
    assert method.before_plan(None, 0) is None


def test_contact_gate_disabled_and_instance_isolation():
    method = SPVLAScheduling(SPVLAConfig.parse({}))
    other = SPVLAScheduling(SPVLAConfig.parse({"enabled": False}))
    action = np.array([0.01, 0.01, 0.4, 0, 0, 0, 1.0])
    for _ in range(6):
        method.after_action(action, 0)
    assert method.before_plan(None, 6) is None
    assert not other.history
    for _ in range(6):
        other.after_action(np.array([0.1, 0.1, 0, 0, 0, 0, 1.0]), 0)
    assert other.before_plan(None, 6) is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"history_size": True},
        {"min_model_actions": 1},
        {"gate_steps": 7},
        {"ridge": 0},
        {"max_z": float("nan")},
        {"enabled": 1},
        {"unknown": 1},
    ],
)
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        SPVLAConfig.parse(parameters)
