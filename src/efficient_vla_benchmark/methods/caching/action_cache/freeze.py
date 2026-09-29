"""Validate independent native population shards and create one immutable bank."""

import argparse
import json
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from .config import ActionCacheConfig


def freeze(shards, output, standard_states):
    """standard_states maps (suite, task) to ALL fifty held-out initial states."""
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    records, sources, entries = [], [], []
    identities, initial_hashes = set(), set()
    checkpoint, sampler = None, None
    capture_schema, replay_digest = None, None
    seconds = []
    for directory in shards:
        metadata = json.loads((directory / "metadata.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(s) for s in (directory / "episodes.jsonl").read_text().splitlines()]
        if metadata.get("kind") != "independent_native_action_cache_population":
            raise ValueError("Population is not a native independent collection")
        schema = metadata.get("capture_schema", 1)
        if schema not in (1, 2) or (capture_schema is not None and capture_schema != schema):
            raise ValueError("Mixed or unsupported population capture schema")
        if schema == 2:
            if metadata.get("clean_action_source") != "native_final_ddim_sample":
                raise ValueError("Clean actions lack native endpoint provenance")
            attempts = [
                json.loads(s) for s in (directory / "attempts.jsonl").read_text().splitlines()
            ]
            fields = ("suite", "task_id", "episode_id", "seed", "initial_state_sha256")
            if [{k: r[k] for k in fields} for r in rows] != attempts:
                raise ValueError("Population attempts differ from completed records")
            digest = metadata.get("replay_population_sha256")
            if capture_schema is not None and digest != replay_digest:
                raise ValueError("Mixed frozen replay populations")
            replay_digest = digest
        capture_schema = schema
        actual = metadata["backend"]["checkpoint_sha256"]
        if checkpoint is not None and (
            checkpoint != actual or sampler != metadata["sampler_parameters"]
        ):
            raise ValueError("Mixed population checkpoint or sampler")
        checkpoint, sampler = actual, metadata["sampler_parameters"]
        sources.append(metadata["source"])
        seconds.append(summary["wall_seconds"])
        with np.load(directory / "bank.npz", allow_pickle=False) as data:
            keys, values, partitions = data["keys"], data["states"], data["suites"]
            clean = data["clean_actions"] if "clean_actions" in data else None
        if schema == 2 and (
            clean is None
            or clean.shape != (len(keys), 16, 7)
            or clean.dtype != np.float32
            or not np.isfinite(clean).all()
        ):
            raise ValueError("Invalid native clean action geometry or values")
        if (
            len(rows) != summary["episodes"]
            or sum(r["success"] for r in rows) != summary["successes"]
            or len(keys) != summary["pairs"]
            or len(keys) != len(values)
            or len(keys) != len(partitions)
        ):
            raise ValueError("Population summary disagrees with its records")
        offset = 0
        for row in rows:
            identity = (row["suite"], row["task_id"], row["episode_id"])
            if identity in identities or row["status"] != "completed":
                raise ValueError("Duplicate or incomplete population episode")
            suite_index = suites.index(row["suite"])
            task, replicate = row["task_id"], row["episode_id"] - 1000
            if not 0 <= task < 10 or replicate not in (0, 1):
                raise ValueError("Population exceeds two episodes per task")
            if row["seed"] != 910000 + replicate * 1000 + suite_index * 10 + task:
                raise ValueError("Unexpected independent population seed")
            flat = suite_index * 10 + task
            with np.load(directory / f"initial-{flat}-{replicate}.npz", allow_pickle=False) as data:
                state = data["state"]
            old = np.asarray(standard_states[(row["suite"], task)])
            digest = stable_hash(state.tolist())
            if (
                old.shape != (50, *state.shape)
                or not np.isfinite(state).all()
                or digest != row["initial_state_sha256"]
                or row["excluded_standard_state_count"] != 50
                or digest in initial_hashes
                or any(np.allclose(state[1:], v[1:], rtol=0, atol=1e-8) for v in old)
            ):
                raise ValueError("Cannot certify population/evaluation state separation")
            initial_hashes.add(digest)
            identities.add(identity)
            count = row["retained_pairs"]
            if count != (row["pending_pairs"] if row["success"] else 0):
                raise ValueError("Failed trajectories must not enter the cache")
            if count:
                end = offset + count
                if end > len(keys) or not np.all(partitions[offset:end] == suite_index):
                    raise ValueError("Invalid population partition or pair count")
                entries.append(
                    (
                        identity,
                        keys[offset:end],
                        values[offset:end],
                        partitions[offset:end],
                        clean[offset:end] if clean is not None else None,
                    )
                )
                offset = end
            records.append(row)
        if offset != len(keys):
            raise ValueError("Unaccounted cache entries")
    expected = {(s, t, 1000 + r) for s in suites for t in range(10) for r in range(2)}
    if identities != expected:
        raise ValueError("Freeze requires exactly the authorized eighty episodes")
    records.sort(key=lambda r: (suites.index(r["suite"]), r["task_id"], r["episode_id"]))
    entries.sort(key=lambda item: (suites.index(item[0][0]), item[0][1], item[0][2]))
    keys = np.concatenate([e[1] for e in entries])
    states = np.concatenate([e[2] for e in entries])
    partitions = np.concatenate([e[3] for e in entries])
    origins = np.concatenate([np.full(len(e[1]), i) for i, e in enumerate(entries)])
    cfg = ActionCacheConfig()
    if (
        keys.shape != (len(keys), cfg.key_dim)
        or states.shape != (len(keys), sampler["num_ddim_steps"], 16, 7)
        or not np.isfinite(keys).all()
        or not np.isfinite(states).all()
        or not np.allclose(np.linalg.norm(keys, axis=1), 1, atol=1e-5)
        or not 0 < len(keys) <= 3000
    ):
        raise ValueError("Invalid bank values/geometry or fixed 3000-entry capacity exceeded")
    profile = {}
    for i, suite in enumerate(suites):
        mask = partitions == i
        local = keys[mask]
        similarity = local @ local.T
        similarity[origins[mask, None] == origins[None, mask]] = -np.inf
        nearest = similarity.max(axis=1)
        profile[suite] = dict(
            entries=int(mask.sum()),
            leave_episode_out_similarity_quantiles=np.quantile(
                nearest, [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
            hit_fractions={
                str(t): float((nearest > t).mean()) for t in (0.9, 0.95, 0.97, 0.98, 0.99, 0.995)
            },
        )
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(keys=keys, states=states, suites=partitions)
    if capture_schema == 2:
        payload["clean_actions"] = np.concatenate([e[4] for e in entries])
    np.savez_compressed(output / "bank.npz", **payload)
    (output / "population.json").write_text(
        json.dumps(dict(episodes=records, sources=sources), indent=2) + "\n"
    )
    (output / "prefill_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    manifest = dict(
        kind="action_cache_cogact_native_success_bank",
        schema_version=capture_schema,
        clean_action_source="native_final_ddim_sample" if capture_schema == 2 else None,
        replay_population_sha256=replay_digest,
        key_dim=cfg.key_dim,
        projection_density=cfg.projection_density,
        projection_seed=cfg.projection_seed,
        checkpoint_sha256=checkpoint,
        sampler_parameters=sampler,
        suites=list(suites),
        episodes_disjoint_from_evaluation=True,
        bank_sha256=digest_file(output / "bank.npz"),
        population_sha256=stable_hash(records),
        population_file_sha256=digest_file(output / "population.json"),
        population_episodes=80,
        successful_episodes=sum(r["success"] for r in records),
        entries=len(keys),
        collection_worker_seconds=sum(seconds),
        collection_longest_shard_seconds=max(seconds),
        collection_cost_excludes_model_loading=True,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    args = parser.parse_args()
    from efficient_vla_benchmark.evaluation.libero.runner import SUITES, configure_libero

    configure_libero(args.libero_root, args.shards[0] / "_private/freeze-libero")
    from libero.libero import benchmark

    standards = {}
    for name in SUITES:
        suite = benchmark.get_benchmark_dict()[name]()
        for task in range(10):
            standards[(name, task)] = suite.get_task_init_states(task)
    print(json.dumps(freeze(args.shards, args.output_dir, standards), indent=2))


if __name__ == "__main__":
    main()
