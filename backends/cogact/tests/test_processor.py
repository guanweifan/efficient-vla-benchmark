import numpy as np
import pytest
from efficient_vla_backend_cogact.processor import prepare_observation, process_actions

from efficient_vla_benchmark.core.types import Observation


def test_action_conversion_is_nonmutating_and_not_euler_conversion():
    original = np.zeros((16, 7))
    original[:, :6] = 0.25
    original[::2, -1] = 1
    snapshot = original.copy()
    result = process_actions(original)
    np.testing.assert_array_equal(original, snapshot)
    np.testing.assert_array_equal(result[:, :6], original[:, :6])
    np.testing.assert_array_equal(result[::2, -1], -1)
    np.testing.assert_array_equal(result[1::2, -1], 1)


@pytest.mark.parametrize(
    "actions", [np.zeros((8, 7)), np.full((16, 7), np.nan), np.full((16, 7), 0.5)]
)
def test_invalid_actions(actions):
    with pytest.raises(ValueError):
        process_actions(actions)


def test_observation_uses_agentview_without_extra_flip():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    obs = Observation(image, np.zeros_like(image), np.zeros(8), "pick up the cup")
    prepared = prepare_observation(obs)
    np.testing.assert_array_equal(np.asarray(prepared["image"]), image)
    assert prepared["instruction"] == obs.instruction
    assert prepared["observation"] is obs
