"""Fixed native controls change one declared variable, never a paper method."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.lifecycle import MethodHost
from efficient_vla_benchmark.core.types import EpisodeContext, Observation


@pytest.mark.parametrize(
    "preset,horizon,steps",
    [("native_h16", 16, 10), ("native_ddim4", 8, 4), ("native_ddim5", 8, 5)],
)
def test_native_control_preset_has_only_its_declared_difference(preset, horizon, steps):
    root = Path(__file__).resolve().parents[1] / "configs/experiments/cogact"
    baseline = asdict(ExperimentConfig.load(root / "baseline.yaml"))
    actual = asdict(ExperimentConfig.load(root / f"{preset}.yaml"))
    expected = {**baseline, "execution_horizon": horizon}
    expected["backend_parameters"] = {**baseline["backend_parameters"], "num_ddim_steps": steps}
    assert actual == expected
    assert actual["method"] == "none" and not actual["parameters"]


class NativeHost(MethodHost):
    def __init__(self):
        self.initialize_method_state()
        self.calls = 0

    def reset(self, context):
        self.calls = 0

    def predict_chunk(self, observation):
        self.calls += 1
        return np.arange(16 * 7).reshape(16, 7)


@pytest.mark.parametrize("horizon", [8, 16])
def test_fixed_prefix_queries_actions_and_reset(horizon):
    host = NativeHost()
    policy = ChunkPolicy(host, horizon)
    context = EpisodeContext("suite", 0, 0)
    observation = Observation(
        np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), np.zeros(8), "task"
    )
    policy.reset(context)
    decisions = [policy.step(observation) for _ in range(33)]
    assert [i for i, d in enumerate(decisions) if d.queried] == list(range(0, 33, horizon))
    assert host.calls == policy.query_count == (33 + horizon - 1) // horizon
    for i, decision in enumerate(decisions):
        np.testing.assert_array_equal(decision.action, np.arange(7) + 7 * (i % horizon))
        assert decision.diagnostics["selected_execution_horizon"] == horizon
    policy.reset(context)
    assert policy.step(observation).queried and host.calls == 1
    assert policy.step(replace(observation, instruction="changed")).queried
    assert host.method is None
