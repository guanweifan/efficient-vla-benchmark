"""Build an immutable timing workload from complete native reference rollouts."""

import argparse
import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.evaluation.libero.aggregate import merge_shards
from efficient_vla_benchmark.evaluation.libero.manifest import load_manifest
from efficient_vla_benchmark.profiling.workload import make_workload


def build(manifest, run_dirs):
    aggregate = merge_shards(manifest, run_dirs)
    if not aggregate["complete"] or aggregate["config"].get("method") != "none":
        raise ValueError("Timing reference must be a complete, error-free native rollout")
    paths = [path for run in run_dirs for path in (run / "traces").glob("*.npz")]
    workload = make_workload(manifest, paths, aggregate["backend"])
    outcomes = {
        (r["suite"], r["task_id"], r["episode_id"]): r for r in aggregate["episode_results"]
    }
    for trace in workload["traces"]:
        context = trace["context"]
        key = (context["suite"], context["task_id"], context["episode_id"])
        if trace["steps"] != outcomes[key]["steps"]:
            raise ValueError("Timing trace was truncated relative to its reference rollout")
    workload["reference_evidence"] = {
        "experiment_id": aggregate["experiment_id"],
        "shards": aggregate["shards"],
        "config": aggregate["config"],
        "episode_results": aggregate["episode_results"],
    }
    workload["sha256"] = stable_hash({k: v for k, v in workload.items() if k != "sha256"})
    return workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen timing workload")
    value = build(load_manifest(args.manifest), args.run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tasks": len(value["traces"]), "sha256": value["sha256"]}))


if __name__ == "__main__":
    main()
