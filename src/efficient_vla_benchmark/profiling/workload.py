"""Fixed task identities and macro latency, independent of models and scheduling."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
from efficient_vla_benchmark.evaluation.libero.manifest import episode_key, validate_manifest
from efficient_vla_benchmark.evaluation.traces import load_trace


def validate_workload(workload):
    if workload.get("sha256") != stable_hash({k: v for k, v in workload.items() if k != "sha256"}):
        raise ValueError("Timing workload digest mismatch")
    manifest = validate_manifest(workload["episode_manifest"])
    if workload.get("schema_version") == 2:
        return validate_compact_workload(workload, manifest)
    if workload.get("schema_version") != 1 or manifest["kind"] != "timing_reference":
        raise ValueError("Expected the fixed forty-task timing workload")
    rows = workload["traces"]
    expected = {(e["suite"], e["task_id"]): e for e in manifest["episodes"]}
    if len(rows) != 40 or len({r["name"] for r in rows}) != 40:
        raise ValueError("Timing workload requires forty unique trace names")
    if {(r["context"]["suite"], r["context"]["task_id"]) for r in rows} != set(expected):
        raise ValueError("Timing workload task coverage differs from manifest")
    for row in rows:
        context = row["context"]
        entry = expected[context["suite"], context["task_id"]]
        if episode_key(context) != episode_key(entry) or context["seed"] != entry["seed"]:
            raise ValueError("Timing workload episode differs from frozen selection")
        if row["steps"] < 1 or "/" in row["name"] or "\\" in row["name"]:
            raise ValueError("Invalid timing trace name or length")
    if workload["length_rule"] != "native_reference_until_success_or_official_limit":
        raise ValueError("Unrecognized reference trace length rule")
    return workload


def validate_compact_workload(workload, manifest):
    """Keep four deterministic prefixes; content-dependent work needs real history."""
    if manifest["kind"] != "timing_reference":
        raise ValueError("Compact replay needs a frozen native reference manifest")
    if workload.get("protocol_id") != "libero_replay4x128_v1":
        raise ValueError("Unknown compact timing protocol")
    if workload.get("length_rule") != "first_min_128_native_control_steps":
        raise ValueError("Unknown compact timing length rule")
    expected = {}
    for entry in manifest["episodes"]:
        suite = entry["suite"]
        if suite not in expected or entry["task_id"] < expected[suite]["task_id"]:
            expected[suite] = entry
    rows = workload["traces"]
    if len(rows) != 4 or len({r["name"] for r in rows}) != 4:
        raise ValueError("Compact replay requires four unique traces")
    if {r["context"]["suite"] for r in rows} != set(expected):
        raise ValueError("Compact replay requires one trace per suite")
    for row in rows:
        context = row["context"]
        entry = expected[context["suite"]]
        if episode_key(context) != episode_key(entry) or context["seed"] != entry["seed"]:
            raise ValueError("Compact replay episode differs from deterministic selection")
        if (
            type(row["source_steps"]) is not int
            or row["source_steps"] < 1
            or row["steps"] != min(128, row["source_steps"])
            or "/" in row["name"]
            or "\\" in row["name"]
        ):
            raise ValueError("Invalid compact trace prefix")
    return workload


def compact_workload(reference):
    """Derive a portable prefix specification without copying or altering observations."""
    validate_workload(reference)
    if reference["schema_version"] != 1:
        raise ValueError("Derive compact timing from the native reference workload")
    chosen = {}
    for row in reference["traces"]:
        suite = row["context"]["suite"]
        if suite not in chosen or row["context"]["task_id"] < chosen[suite]["context"]["task_id"]:
            chosen[suite] = row
    traces = []
    for suite in sorted(chosen):
        row = deepcopy(chosen[suite])
        row["source_steps"] = row["steps"]
        row["steps"] = min(128, row["steps"])
        traces.append(row)
    value = {
        "schema_version": 2,
        "protocol_id": "libero_replay4x128_v1",
        "episode_manifest": deepcopy(reference["episode_manifest"]),
        "reference_backend": deepcopy(reference["reference_backend"]),
        "reference_workload_sha256": reference["sha256"],
        "length_rule": "first_min_128_native_control_steps",
        "traces": traces,
    }
    value["sha256"] = stable_hash(value)
    return validate_workload(value)


def workload_summary(rows, workload, repeats):
    validated = task_macro_summary(rows, workload["traces"], repeats)
    if workload["schema_version"] == 1:
        return "task_macro", validated
    if repeats != 1:
        raise ValueError("Compact protocol requires exactly one measured pass")
    return "control_workload", {
        "aggregation": "fixed_control_mean",
        "mean_ms": float(np.mean([r["latency_ms"] for r in rows])),
        "control_steps": len(rows),
        "traces": len(workload["traces"]),
        "per_trace": validated["per_task"],
    }


def make_workload(manifest, trace_paths, baseline):
    validate_manifest(manifest)
    if manifest["kind"] != "timing_reference" or baseline.get("method") != "none":
        raise ValueError("A timing workload requires timing40 and a native reference")
    traces = []
    for path in sorted(trace_paths):
        context, samples, steps = load_trace(path, require_control_steps=True)
        traces.append(
            {
                "name": path.name,
                "sha256": digest_file(path),
                "context": asdict(context),
                "steps": len(steps),
                "depth": samples[0].depth is not None,
            }
        )
    result = {
        "schema_version": 1,
        "episode_manifest": manifest,
        "reference_backend": baseline,
        "length_rule": "native_reference_until_success_or_official_limit",
        "traces": traces,
    }
    result["sha256"] = stable_hash(result)
    return validate_workload(result)


def verify_trace_files(workload, paths, backend):
    validate_workload(workload)
    by_name = {path.name: path for path in paths}
    if len(by_name) != len(paths) or set(by_name) != {r["name"] for r in workload["traces"]}:
        raise ValueError("Supplied traces do not exactly cover the timing workload")
    for key in ("backend", "model_variant", "checkpoint_sha256"):
        if backend.get(key) != workload["reference_backend"].get(key):
            raise ValueError(f"Timing workload baseline differs in {key}")
    for row in workload["traces"]:
        if digest_file(by_name[row["name"]]) != row["sha256"]:
            raise ValueError("Timing trace content differs from fixed workload")


def task_macro_summary(rows, traces, repeats):
    """Average controls within each task, then average tasks; never average speedups."""
    if len({r["name"] for r in traces}) != len(traces) or not traces:
        raise ValueError("Task latency requires unique trace identities")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["repeat"], row["trace"]].append(row)
    expected = {(repeat, trace["name"]) for repeat in range(repeats) for trace in traces}
    if set(grouped) != expected:
        raise ValueError("Missing or foreign repeat/task timing rows")
    per_task = []
    for trace in traces:
        means = []
        for repeat in range(repeats):
            samples = grouped[repeat, trace["name"]]
            if [r["control_step"] for r in samples] != list(range(trace["steps"])):
                raise ValueError("Task timing must contain each reference control exactly once")
            values = [r["latency_ms"] for r in samples]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("Invalid task latency")
            means.append(float(np.mean(values)))
        per_task.append(
            {
                "trace": trace["name"],
                "context": trace["context"],
                "controls_per_repeat": trace["steps"],
                "repeat_mean_ms": means,
                "mean_ms": float(np.mean(means)),
            }
        )
    repeat_means = [
        float(np.mean([t["repeat_mean_ms"][i] for t in per_task])) for i in range(repeats)
    ]
    return {
        "aggregation": "task_macro_mean_of_control_latency",
        "tasks": len(per_task),
        "mean_ms": float(np.mean(repeat_means)),
        "repeat_mean_ms": repeat_means,
        "repeat_std_ms": float(np.std(repeat_means)),
        "per_task": per_task,
    }
