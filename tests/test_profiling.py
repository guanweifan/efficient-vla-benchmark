import numpy as np
import pytest

from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.types import EpisodeContext, Observation
from efficient_vla_benchmark.evaluation.traces import save_trace
from efficient_vla_benchmark.profiling.benchmark import benchmark, latency_summary


def test_latency_statistics():
    result = latency_summary([1.0, 2.0, 3.0])
    assert result["mean_ms"] == 2.0
    assert result["p50_ms"] == 2.0
    assert result["p95_ms"] == pytest.approx(2.9)
    assert result["count"] == 3


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")]])
def test_invalid_latency_samples(values):
    with pytest.raises(ValueError):
        latency_summary(values)


def test_control_replay_counts_replanning_and_all_frames(tmp_path):
    # Contract test only: these CPU stub counters are not performance evidence.
    class Monitor:
        def synchronize(self):
            pass

        def reset_peak_memory(self):
            pass

        def peak_memory(self):
            return {"allocated_bytes": None, "reserved_bytes": None}

        def hardware(self):
            return {"runtime": "unit_test_stub"}

    class Backend:
        runtime = Monitor()
        query_diagnostics = {}

        def metadata(self):
            return {"requires_control_history": True, "action_dim": 7}

        def reset(self, context):
            self.steps = 0

        def observe(self, obs):
            self.steps += 1

        def after_action(self, action, remaining):
            return self.steps == 2

        def predict_chunk(self, obs):
            return np.zeros((8, 7))

    obs = Observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(8),
        "test",
    )
    trace = tmp_path / "trace.npz"
    save_trace(trace, EpisodeContext("libero_goal", 0, 0), [obs] * 4, list(range(4)))
    config = ExperimentConfig("openvla_oft", "none", "bfloat16", 8, {})
    backend = Backend()
    result = benchmark(backend, config, [trace], tmp_path / "output", repeats=1, warmup=1)
    assert result["all_queries"]["count"] == 2
    assert result["all_control_steps"]["count"] == backend.steps == 4
    assert result["query_fraction"] == 0.5
    assert result["reference_replay_not_closed_loop"]
    assert result["peak_allocated_bytes"] is None
    with pytest.raises(ValueError, match="complete mechanism"):
        benchmark(backend, config, [trace], tmp_path / "invalid", mode="query")


def test_replacements_are_timed_but_never_counted_as_queries(tmp_path):
    from types import SimpleNamespace

    from efficient_vla_benchmark.core.lifecycle import MethodHost
    from efficient_vla_benchmark.core.method import InferenceMethod

    class Replacement(InferenceMethod):
        def before_plan(self, observation, step):
            return np.ones(7) if step == 2 else None

    class Backend(MethodHost):
        runtime = SimpleNamespace(
            synchronize=lambda: None,
            reset_peak_memory=lambda: None,
            peak_memory=lambda: {"allocated_bytes": None, "reserved_bytes": None},
            hardware=lambda: {"runtime": "unit_test_stub"},
        )

        def __init__(self):
            self.initialize_method_state()
            self.attach_method(Replacement())

        def reset(self, context):
            self.method.reset(context)

        def metadata(self):
            return {"requires_control_history": True, "action_dim": 7}

        def predict_chunk(self, obs):
            return np.zeros((2, 7))

    obs = Observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(8),
        "test",
    )
    trace = tmp_path / "trace.npz"
    save_trace(trace, EpisodeContext("libero_goal", 0, 0), [obs] * 4, list(range(4)))
    result = benchmark(
        Backend(),
        ExperimentConfig("fixture", execution_horizon=2),
        [trace],
        tmp_path / "output",
        repeats=1,
        warmup=1,
    )
    assert result["all_queries"]["count"] == 2
    assert result["replacement_steps"] == 1
    assert result["replacement_latency"]["count"] == 1
    assert result["all_control_steps"]["count"] == 4
