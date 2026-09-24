"""Plan portable development jobs and strictly aggregate their episode evidence."""

import argparse
import json
from pathlib import Path

import yaml

from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.evaluation.libero.aggregate import merge_shards
from efficient_vla_benchmark.evaluation.libero.manifest import load_manifest, select_shard


def plan(matrix, manifest, campaign, shards, baselines_only=False):
    jobs = []
    for backend, presets in matrix.items():
        if len(presets) != len(set(presets)):
            raise ValueError("Duplicate experiment preset")
        for preset in presets:
            path = f"configs/experiments/{backend}/{preset}.yaml"
            config = ExperimentConfig.load(Path(path))
            if config.backend != backend:
                raise ValueError("Preset/backend mismatch")
            if baselines_only and config.method != "none":
                continue
            for index in range(shards):
                select_shard(manifest, index, shards)
                jobs.append(
                    {
                        "id": f"{backend}--{preset}--s{index:03d}",
                        "backend": backend,
                        "preset": preset,
                        "config": path,
                        "shard_index": index,
                        "num_shards": shards,
                        "artifact": config.method == "awq",
                    }
                )
    # Matched cadence baselines first; expensive H1 and auxiliary cases precede ordinary ones.
    jobs.sort(
        key=lambda j: (
            not j["preset"].startswith("baseline"),
            j["preset"] not in {"vote_ensemble", "actuation_slack_refresh_deletion"},
            j["shard_index"],
            j["backend"],
            j["preset"],
        )
    )
    return {
        "schema_version": 1,
        "campaign": campaign,
        "manifest": f"outputs/{campaign}/episodes.json",
        "manifest_sha256": manifest["sha256"],
        "closed": True,
        "jobs": jobs,
    }


def progress_summary(value):
    if "jobs" in value:
        return {"jobs": len(value["jobs"])}
    rows = value["configurations"]
    return {
        "configurations": len(rows),
        "completed_configurations": sum(bool(r["complete"]) for r in rows),
        "episodes": sum(r["episodes"] for r in rows),
        "errors": sum(r.get("errors", 0) for r in rows),
        "all_complete": value["all_complete"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    create = subs.add_parser("plan")
    create.add_argument("--matrix", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--campaign", required=True)
    create.add_argument("--shards", type=int, default=8)
    create.add_argument("--baselines-only", action="store_true")
    create.add_argument("--output", type=Path, required=True)
    merge = subs.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--jobs", type=Path, required=True)
    merge.add_argument("--runs", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "plan":
        value = plan(
            yaml.safe_load(args.matrix.read_text()),
            manifest,
            args.campaign,
            args.shards,
            args.baselines_only,
        )
    else:
        jobs = json.loads(args.jobs.read_text())["jobs"]
        groups = {}
        for job in jobs:
            groups.setdefault((job["backend"], job["preset"]), []).append(job)
        results = []
        for (backend, preset), members in groups.items():
            paths = [
                args.runs / j["id"]
                for j in members
                if (args.runs / j["id"] / "episodes.jsonl").is_file()
            ]
            if paths:
                result = merge_shards(manifest, paths)
                result["preset"] = preset
                results.append(result)
            else:
                results.append(
                    {
                        "backend_id": backend,
                        "preset": preset,
                        "complete": False,
                        "episodes": 0,
                        "success_rate": None,
                    }
                )
        value = {
            "kind": "development_validation",
            "manifest_sha256": manifest["sha256"],
            "configurations": results,
            "all_complete": all(r["complete"] for r in results),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_name": args.output.name,
                **progress_summary(value),
            }
        )
    )


if __name__ == "__main__":
    main()
