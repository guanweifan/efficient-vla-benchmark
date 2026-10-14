import numpy as np
import pytest

from efficient_vla_benchmark.calibration import libero
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


def test_chunk_sampler_preserves_raw_actions_and_full_horizon(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    class Steps(list):
        def cardinality(self):
            return Tensor(len(self))

        def skip(self, n):
            return Steps(self[n:])

        def take(self, n):
            return Steps(self[:n])

    steps = Steps(
        [
            {
                "observation": {
                    "image": Tensor(np.full((2, 2, 3), i, np.uint8)),
                    "wrist_image": Tensor(np.zeros((2, 2, 3), np.uint8)),
                    "state": Tensor(np.zeros(8, np.float32)),
                },
                "language_instruction": Tensor(b"test"),
                "action": Tensor(np.full(7, i, np.float32)),
            }
            for i in range(12)
        ]
    )
    builder = SimpleNamespace(
        info=SimpleNamespace(splits={"train": SimpleNamespace(num_examples=1)}),
        as_dataset=lambda **kwargs: [{"steps": steps}],
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            config=SimpleNamespace(set_visible_devices=lambda *args: None),
            data=SimpleNamespace(Options=lambda: SimpleNamespace(threading=SimpleNamespace())),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorflow_datasets",
        SimpleNamespace(
            builder_from_directory=lambda path: builder, ReadConfig=lambda **kwargs: kwargs
        ),
    )
    for suite in libero.SUITES:
        root = tmp_path / f"{suite}_no_noops/1.0.0"
        root.mkdir(parents=True)
        (root / "dataset_info.json").write_text(json.dumps({"test": True}))
    values = list(libero.training_action_chunks(tmp_path, 4, 0, 8))
    assert len(values) == 4
    for context, observation, identity, actions in values:
        start = identity["step_index"]
        assert 0 <= start <= 4 and identity["action_horizon"] == 8
        np.testing.assert_array_equal(actions[:, 0], np.arange(start, start + 8))
        assert observation.image[0, 0, 0] == start
        assert identity["split"] == "train"
    observations = list(libero.training_observations(tmp_path, 4, 0))
    assert all(len(x) == 3 and "action_horizon" not in x[2] for x in observations)
    with pytest.raises(ValueError, match="complete chunk"):
        list(libero.training_action_chunks(tmp_path, 4, 0, 13))
