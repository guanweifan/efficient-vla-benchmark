"""Strict union of independently executed shards, with explicit incomplete outcomes."""

import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from .manifest import SUITES, episode_key, select_shard, validate_manifest


def merge_shards(manifest, run_dirs):
    validate_manifest(manifest)
    expected = {episode_key(e): e for e in manifest["episodes"]}
    records, evidence, identity = {}, [], None
    reference = None
    for directory in sorted(map(Path, run_dirs)):
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["kind"] != manifest["kind"]:
            raise ValueError("Shard kind differs from its episode manifest")
        if metadata["episode_manifest"]["sha256"] != manifest["sha256"]:
            raise ValueError("Shards use different episode manifests")
        comparable = {
            "experiment_id": metadata["experiment_id"],
        }
        current = stable_hash(comparable)
        if identity is not None and current != identity:
            raise ValueError("Shards differ in policy, code, protocol or runtime")
        identity, reference = current, metadata
        shard = metadata["shard"]
        allowed = select_shard(manifest, shard["index"], shard["count"])
        if shard["episodes"] != allowed:
            raise ValueError("Shard declaration differs from manifest")
        allowed_keys = {episode_key(e) for e in allowed}
        path = directory / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            key = episode_key(row)
            if key in records:
                raise ValueError("Duplicate episode across shards or attempts")
            if key not in allowed_keys or row["seed"] != expected[key]["seed"]:
                raise ValueError("Episode is outside the declared shard or has the wrong seed")
            if (
                row["status"] not in {"completed", "error"}
                or type(row["success"]) is not bool
                or (row["status"] == "error" and row["success"])
            ):
                raise ValueError("Invalid episode outcome")
            records[key] = row
        evidence.append(
            {
                "source": metadata["source"],
                "run_id": directory.name,
                "shard_index": shard["index"],
                "shard_count": shard["count"],
                "hardware": metadata.get("hardware"),
                "files_sha256": {
                    n: digest_file(directory / n) for n in ("metadata.json", "episodes.jsonl")
                },
            }
        )
    if reference is None:
        raise ValueError("No evaluation shards supplied")
    suites = {}
    for suite in SUITES:
        rows = [r for k, r in records.items() if k[0] == suite]
        successes = sum(r["success"] for r in rows)
        errors = sum(r["status"] == "error" for r in rows)
        expected_suite = 10 * manifest["samples_per_task"]
        complete = len(rows) == expected_suite and errors == 0
        suites[suite] = {
            "episodes": len(rows),
            "expected": expected_suite,
            "successes": successes,
            "errors": errors,
            "success_rate": successes / expected_suite if complete else None,
        }
    successes = sum(r["success"] for r in records.values())
    errors = sum(r["status"] == "error" for r in records.values())
    population = len(expected)
    complete = len(records) == population and errors == 0
    return {
        "schema_version": 1,
        "kind": manifest["kind"],
        "manifest_sha256": manifest["sha256"],
        "experiment_id": reference["experiment_id"],
        "config": reference["config"],
        "backend": reference["backend"],
        "source": reference["source"],
        "expected_episodes": population,
        "episodes": len(records),
        "successes": successes,
        "errors": errors,
        "complete": complete,
        "success_rate": successes / population if complete else None,
        "successes_over_planned": successes / population,
        "formal_success_rate": successes / population
        if complete and manifest["kind"] == "final_evaluation"
        else None,
        "missing": [expected[k] for k in expected if k not in records],
        "suites": suites,
        "episode_results": [records[k] for k in sorted(records)],
        "shards": evidence,
    }
