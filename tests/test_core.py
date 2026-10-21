import numpy as np
import pytest

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.types import EpisodeContext, Observation


def observation():
    return Observation(
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros(8),
        "pick up the object",
    )


class Predictor:
    def reset(self, context):
        self.context = context

    def predict_chunk(self, obs):
        return np.arange(56).reshape(8, 7).astype(np.float32)

    def observe(self, obs):
        pass

    def after_action(self, action, remaining):
        return False


def test_chunk_executes_prefix_and_requeries():
    policy = ChunkPolicy(Predictor(), 2)
    policy.reset(EpisodeContext("libero_spatial", 0, 0))
    first = policy.step(observation())
    second = policy.step(observation())
    third = policy.step(observation())
    np.testing.assert_array_equal(first.action, np.arange(7))
    np.testing.assert_array_equal(second.action, np.arange(7, 14))
    np.testing.assert_array_equal(third.action, first.action)
    assert first.queried and not second.queried and third.queried
    assert policy.query_count == 2


def test_reset_clears_remaining_actions():
    policy = ChunkPolicy(Predictor(), 8)
    context = EpisodeContext("libero_spatial", 0, 0)
    policy.reset(context)
    first = policy.step(observation())
    policy.reset(context)
    np.testing.assert_array_equal(policy.step(observation()).action, first.action)
    assert policy.query_count == 1


@pytest.mark.parametrize("predicted,executed", [(4, 3), (16, 12)])
def test_common_controller_does_not_assume_oft_chunk_length(predicted, executed):
    class VariableChunk(Predictor):
        def predict_chunk(self, obs):
            return np.arange(predicted * 7).reshape(predicted, 7)

    policy = ChunkPolicy(VariableChunk(), executed)
    policy.reset(EpisodeContext("libero_goal", 0, 0))
    decisions = [policy.step(observation()) for _ in range(executed + 1)]
    assert [i for i, item in enumerate(decisions) if item.queried] == [0, executed]
    np.testing.assert_array_equal(decisions[-1].action, np.arange(7))


def test_execution_horizon_cannot_exceed_returned_chunk():
    policy = ChunkPolicy(Predictor(), 9)
    policy.reset(EpisodeContext("libero_goal", 0, 0))
    with pytest.raises(ValueError, match="enough finite"):
        policy.step(observation())


def test_method_controller_receives_every_frame_and_can_replan():
    class Controller(Predictor):
        def reset(self, context):
            self.frames = 0

        def observe(self, obs):
            self.frames += 1

        def after_action(self, action, remaining):
            action[:] = -100  # The emitted command is protected from mutation.
            return self.frames == 2 and remaining > 0

    predictor = Controller()
    policy = ChunkPolicy(predictor, 8)
    context = EpisodeContext("libero_spatial", 0, 0)
    policy.reset(context)
    decisions = [policy.step(observation()) for _ in range(3)]
    assert [d.queried for d in decisions] == [True, False, True]
    assert predictor.frames == 3
    np.testing.assert_array_equal(decisions[1].action, np.arange(7, 14))
    assert decisions[1].diagnostics["replan_requested"]
    policy.reset(context)
    assert predictor.frames == 0


@pytest.mark.parametrize(
    "text",
    [
        "method: [awq, vla_cache]",
        "unknown: 1",
        "execution_horizon: 0",
        "backend: [cogact, openvla_oft]",
        "backend: custom\nparameters: {resource: /private/resource}",
    ],
)
def test_invalid_config_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        ExperimentConfig.load(path)


def test_observation_validation():
    observation().validate()
    obs = observation()
    obs.proprio[0] = np.nan
    with pytest.raises(ValueError):
        obs.validate()


def test_core_does_not_restrict_model_precision_or_horizon(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("backend: custom\nprecision: float32\nexecution_horizon: 32\n")
    assert ExperimentConfig.load(path).execution_horizon == 32


def test_config_composition_and_cycles(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("backend: custom\nparameters: {a: 1, b: 2}\n")
    preset = tmp_path / "run.yaml"
    preset.write_text("extends: base.yaml\nparameters: {b: 3}\n")
    assert ExperimentConfig.load(preset).parameters == {"a": 1, "b": 3}
    base.write_text("extends: run.yaml\nbackend: custom\n")
    with pytest.raises(ValueError, match="Cyclic"):
        ExperimentConfig.load(preset)


def test_control_is_not_tied_to_seven_action_dimensions():
    class DifferentModel(Predictor):
        def predict_chunk(self, obs):
            return np.zeros((16, 12))

    policy = ChunkPolicy(DifferentModel(), 12)
    policy.reset(EpisodeContext("custom", 0, 0))
    assert policy.step(observation()).action.shape == (12,)
