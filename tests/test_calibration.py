import numpy as np
import pytest

from efficient_vla_benchmark.calibration.libero import (
    observation_digest,
    stratified_episode_indices,
)
from efficient_vla_benchmark.core.types import Observation


def test_episode_selection_is_deterministic_unique_and_stratified():
    selected = stratified_episode_indices(432, 8, 0)
    assert selected == stratified_episode_indices(432, 8, 0)
    assert selected == sorted(set(selected))
    edges = np.linspace(0, 432, 9, dtype=int)
    assert all(low <= value < high for value, low, high in zip(selected, edges[:-1], edges[1:]))
    with pytest.raises(ValueError):
        stratified_episode_indices(4, 5, 0)


def test_calibration_identity_tracks_content_without_paths():
    observation = Observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
        "test",
    )
    original = observation_digest(observation)
    observation.proprio[0] = 1
    assert observation_digest(observation) != original
    assert len(original) == 64
