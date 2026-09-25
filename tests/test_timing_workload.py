import copy
from types import SimpleNamespace

import numpy as np
import pytest

from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.core.types import EpisodeContext, Observation
from efficient_vla_benchmark.evaluation.libero.manifest import (
    SUITES,
    derive_manifest,
    make_manifest,
)
from efficient_vla_benchmark.evaluation.traces import save_trace
from efficient_vla_benchmark.profiling.benchmark import benchmark
from efficient_vla_benchmark.profiling.workload import (
    make_workload,
    task_macro_summary,
    validate_workload,
    verify_trace_files,
)


def test_macro_latency_does_not_weight_long_tasks_or_average_speedup_ratios():
    traces = [
        {"name": "short", "context": {}, "steps": 1},
        {"name": "long", "context": {}, "steps": 9},
    ]
    rows = [
        {
            "repeat": i,
            "trace": t["name"],
            "control_step": step,
            "latency_ms": 10.0 if t["name"] == "short" else 2.0,
        }
        for i in range(3)
        for t in traces
        for step in range(t["steps"])
    ]
    result = task_macro_summary(rows, traces, 3)
    assert result["mean_ms"] == 6.0  # control-weighted mean would be 2.8
    assert result["repeat_mean_ms"] == [6.0, 6.0, 6.0]
    for invalid in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError):
            task_macro_summary(invalid, traces, 3)


def test_fixed_workload_rejects_wrong_ids_files_or_backend(tmp_path):
    resources = {
        f"{s}/{t}": {
            "task_name": f"task_{t}",
            "bddl": {"path": f"bddl/{s}/{t}", "sha256": "abc"},
            "init_states": {"path": f"init/{s}/{t}", "sha256": "def"},
        }
        for s in SUITES
        for t in range(10)
    }
    manifest = derive_manifest(make_manifest(resources), "timing")
    obs = Observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(8),
        "test",
        np.ones((2, 2), dtype=np.float32),
    )
    paths = []
    for entry in manifest["episodes"]:
        context = EpisodeContext(
            entry["suite"], entry["task_id"], entry["episode_id"], entry["seed"]
        )
        path = tmp_path / f"{context.suite}-{context.task_id}-{context.episode_id}.npz"
        save_trace(path, context, [obs] * 2, [0, 1])
        paths.append(path)
    baseline = {
        "backend": "fixture",
        "model_variant": "fixture",
        "checkpoint_sha256": "a",
        "method": "none",
    }
    workload = make_workload(manifest, paths, baseline)
    assert len(workload["traces"]) == 40
    verify_trace_files(workload, paths, baseline)
    with pytest.raises(ValueError, match="cover"):
        verify_trace_files(workload, paths[:-1], baseline)
    with pytest.raises(ValueError, match="checkpoint"):
        verify_trace_files(workload, paths, dict(baseline, checkpoint_sha256="b"))
    changed = copy.deepcopy(workload)
    changed["traces"][0]["context"]["episode_id"] = 49
    changed["sha256"] = stable_hash({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="episode"):
        validate_workload(changed)

    class Backend:
        runtime = SimpleNamespace(
            synchronize=lambda: None,
            reset_peak_memory=lambda: None,
            peak_memory=lambda: {"allocated_bytes": None, "reserved_bytes": None},
            hardware=lambda: {"runtime": "unit_test_stub"},
        )

        def metadata(self):
            return dict(baseline, action_dim=7)

        def reset(self, context):
            pass

        def observe(self, observation):
            pass

        def after_action(self, action, remaining):
            return False

        def predict_chunk(self, observation):
            return np.zeros((8, 7))

    result = benchmark(
        Backend(),
        ExperimentConfig("fixture"),
        paths,
        tmp_path / "timing",
        repeats=2,
        warmup=1,
        workload=workload,
    )
    assert result["task_macro"]["tasks"] == 40
    assert result["all_control_steps"]["count"] == 160
    assert len(result["task_macro"]["repeat_mean_ms"]) == 2
    paths[0].write_bytes(b"corrupted fixture")
    with pytest.raises(ValueError, match="content"):
        verify_trace_files(workload, paths, baseline)
