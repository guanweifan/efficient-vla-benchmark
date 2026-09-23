"""Synchronized batch-one timing on real sequential observation traces."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata
from efficient_vla_benchmark.evaluation.traces import load_trace


def latency_summary(values: list[float]) -> dict:
    if not values or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Latency samples must be nonempty, finite and positive")
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def benchmark(
    backend,
    config,
    trace_paths: list[Path],
    output_dir: Path,
    repeats: int = 3,
    warmup: int = 10,
    mode: str = "control",
):
    if (
        type(repeats) is not int
        or type(warmup) is not int
        or repeats < 1
        or warmup < 1
        or mode not in {"control", "query", "prepared"}
    ):
        raise ValueError("Invalid timing protocol")
    monitor = backend.runtime
    if not trace_paths:
        raise ValueError("At least one real observation trace is required")
    if (output_dir / "timings.jsonl").exists() or (output_dir / "metadata.json").exists():
        raise ValueError("Timing output already exists; choose a new output directory")
    if mode != "control" and backend.metadata().get("requires_control_history"):
        raise ValueError("This method requires control replay to preserve its complete mechanism")
    traces = [
        (path.name, *load_trace(path, require_control_steps=mode == "control"))
        for path in trace_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 3,
        "kind": "sequential_policy_timing",
        "backend": backend.metadata(),
        "config": asdict(config),
        "source": source_metadata(),
        "protocol": {
            "mode": mode,
            "repeats": repeats,
            "warmup_queries": warmup,
            "batch_size": 1,
            "device_synchronization": "before_and_after_each_measured_call",
            "input_io_included": False,
            "model_loading_included": False,
            "preprocessing_and_transfer_included": mode != "prepared",
            "postprocessing_included": True,
            "episode_state_reset": True,
            "controller_included": mode == "control",
            "observations": "fixed_reference_trajectory_not_closed_loop",
            "query_definition": "served_native_call; auxiliary native calls counted separately",
        },
        "hardware": monitor.hardware(),
        "traces": [{"name": path.name, "sha256": digest_file(path)} for path in trace_paths],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    # Warm kernels separately; their method history is discarded before measurement.
    _, first_context, first_samples, _ = traces[0]
    policy = ChunkPolicy(backend, config.execution_horizon)
    policy.reset(first_context)
    if mode == "control":
        index = queries = 0
        while queries < warmup:
            if index and index % len(first_samples) == 0:
                policy.reset(first_context)
            decision = policy.step(first_samples[index % len(first_samples)])
            queries += int(decision.queried)
            index += 1
    else:
        for index in range(warmup):
            backend.predict_chunk(first_samples[index % len(first_samples)])
    monitor.synchronize()

    rows = []
    peak_allocated = None
    peak_reserved = None
    with (output_dir / "timings.jsonl").open("x") as stream:
        for repeat in range(repeats):
            for name, context, samples, steps in traces:
                policy.reset(context)
                prepared = (
                    [backend.prepare(sample) for sample in samples] if mode == "prepared" else None
                )
                monitor.synchronize()
                monitor.reset_peak_memory()
                query_index = 0
                for index, sample in enumerate(samples):
                    monitor.synchronize()
                    start = time.perf_counter_ns()
                    if mode == "control":
                        decision = policy.step(sample)
                        actions, queried = decision.action, decision.queried
                        diagnostics = dict(decision.diagnostics)
                    else:
                        actions = (
                            backend.predict_chunk(sample)
                            if prepared is None
                            else backend.forward(prepared[index])
                        )
                        queried, diagnostics = True, {}
                    monitor.synchronize()
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    expected_shape = (
                        (metadata["backend"]["action_dim"],)
                        if mode == "control"
                        else (
                            metadata["backend"]["predicted_horizon"],
                            metadata["backend"]["action_dim"],
                        )
                    )
                    if actions.shape != expected_shape or not np.isfinite(actions).all():
                        raise ValueError("Invalid actions during timing")
                    if queried:
                        diagnostics.update(getattr(backend, "query_diagnostics", {}))
                    row = {
                        "repeat": repeat,
                        "trace": name,
                        "query_index": query_index if queried else None,
                        "queried": queried,
                        "control_step": int(steps[index]),
                        "cache_cold": queried and query_index == 0,
                        "latency_ms": elapsed_ms,
                        "diagnostics": diagnostics,
                    }
                    query_index += int(queried)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                stream.flush()
                memory = monitor.peak_memory()
                if memory["allocated_bytes"] is not None:
                    peak_allocated = max(peak_allocated or 0, memory["allocated_bytes"])
                if memory["reserved_bytes"] is not None:
                    peak_reserved = max(peak_reserved or 0, memory["reserved_bytes"])
                del prepared
                print(
                    json.dumps({"repeat": repeat, "trace": name, "queries": query_index}),
                    flush=True,
                )
    query_rows = [row for row in rows if row["queried"]]
    all_latencies = [row["latency_ms"] for row in query_rows]
    steady = [row["latency_ms"] for row in query_rows if not row["cache_cold"]]
    result = {
        "mode": mode,
        "all_queries": latency_summary(all_latencies),
        "episode_first_query": latency_summary(
            [row["latency_ms"] for row in query_rows if row["cache_cold"]]
        ),
        "steady_state": latency_summary(steady) if steady else None,
        "per_repeat": [
            latency_summary([row["latency_ms"] for row in query_rows if row["repeat"] == index])
            for index in range(repeats)
        ],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "formal_success_rate_measured": False,
        "reference_replay_not_closed_loop": True,
        "auxiliary_model_calls": sum(
            row["diagnostics"].get("auxiliary_queries", 0) for row in query_rows
        ),
        "total_native_model_calls": len(query_rows) + sum(
            row["diagnostics"].get("auxiliary_queries", 0) for row in query_rows
        ),
    }
    if mode == "control":
        result["all_control_steps"] = latency_summary([row["latency_ms"] for row in rows])
        result["replay_policy_time_ms"] = sum(row["latency_ms"] for row in rows)
        result["query_fraction"] = len(query_rows) / len(rows)
        replacement_rows = [
            row for row in rows if row["diagnostics"].get("action_source") == "replacement"
        ]
        result["replacement_steps"] = len(replacement_rows)
        result["replacement_latency"] = (
            latency_summary([row["latency_ms"] for row in replacement_rows])
            if replacement_rows
            else None
        )
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2), flush=True)
    return result
