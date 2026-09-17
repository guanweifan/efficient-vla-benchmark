"""Runs only in the dedicated environment with both external templates installed."""

import importlib.util

import numpy as np
import pytest

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.registry import backend_entries, build_backend, method_catalog
from efficient_vla_benchmark.core.types import EpisodeContext, Observation
from efficient_vla_benchmark.evaluation.traces import save_trace
from efficient_vla_benchmark.profiling.benchmark import benchmark


def test_independent_plugin_without_model_dependencies(tmp_path):
    if "example" not in backend_entries() or "example_scale" not in method_catalog():
        pytest.skip("Install the two external templates in a dedicated CPU environment")
    for name in ("torch", "transformers", "tensorflow", "prismatic", "libero"):
        assert importlib.util.find_spec(name) is None
    config = ExperimentConfig(
        "example",
        "example_scale",
        execution_horizon=16,
        model_variant="long_chunk",
        parameters={"scale": 0.25},
    )
    backend = build_backend(config, tmp_path, "cpu")
    context = EpisodeContext("fixture", 0, 0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = Observation(image, image.copy(), np.zeros(11), "contract test")
    policy = ChunkPolicy(backend, config.execution_horizon)
    policy.reset(context)
    decisions = [policy.step(observation) for _ in range(17)]
    assert sum(decision.queried for decision in decisions) == 2
    np.testing.assert_array_equal(decisions[0].action, [0.25] * 3)
    trace = tmp_path / "trace.npz"
    save_trace(trace, context, [observation] * 17, list(range(17)))
    result = benchmark(backend, config, [trace], tmp_path / "profile", repeats=1, warmup=1)
    assert result["all_queries"]["count"] == 2
    assert result["all_control_steps"]["count"] == 17
    assert result["peak_allocated_bytes"] is None
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observation), np.ones((32, 3)))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.predict_chunk(observation)
