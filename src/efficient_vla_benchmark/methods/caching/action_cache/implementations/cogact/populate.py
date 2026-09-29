"""Collect at most two independent native episodes per task; no test labels."""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.provenance import source_metadata, stable_hash
from efficient_vla_benchmark.core.types import EpisodeContext
from efficient_vla_benchmark.evaluation.libero.runner import SUITES, configure_libero, run_episode

from ...config import ActionCacheConfig


def replay_states(directories):
    """Validate exactly the previous frozen identities, including unsuccessful ones."""
    if not directories:
        return {}, None
    from efficient_vla_benchmark.core.provenance import digest_file

    states, sources = {}, []
    for directory in directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("kind") != "independent_native_action_cache_population":
            raise ValueError("Replay inputs are not independent native population shards")
        sources.append(digest_file(directory / "episodes.jsonl"))
        for line in (directory / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            flat = SUITES.index(row["suite"]) * 10 + row["task_id"]
            replicate = row["episode_id"] - 1000
            identity = (flat, replicate)
            if identity in states or not 0 <= flat < 40 or replicate not in (0, 1):
                raise ValueError("Duplicate or invalid frozen population identity")
            if row["status"] != "completed" or row["seed"] != 910000 + replicate * 1000 + flat:
                raise ValueError("Incomplete or unexpected frozen population record")
            with np.load(directory / f"initial-{flat}-{replicate}.npz", allow_pickle=False) as data:
                state = data["state"].copy()
            if (
                not np.isfinite(state).all()
                or stable_hash(state.tolist()) != row["initial_state_sha256"]
            ):
                raise ValueError("Frozen replay state checksum mismatch")
            states[identity] = state
    if set(states) != {(f, r) for f in range(40) for r in range(2)}:
        raise ValueError("Replay requires all eighty original independent identities")
    return states, stable_hash(sorted(sources))


def collect(model_path, libero_root, output, shard, shards, replay_population=None):
    if not 1 <= shards <= 40 or not 0 <= shard < shards:
        raise ValueError("Invalid independent population shard")
    if output.exists():
        raise ValueError("Population output exists; inspect it instead of replacing or resampling")
    frozen_states, replay_digest = replay_states(replay_population)
    configure_libero(libero_root, output / "_private/libero")
    from efficient_vla_backend_cogact.backend import CogACTBackend
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from .capture import NativeActionCapture

    host = CogACTBackend(model_path)
    keys, states, clean_actions, partitions, records = [], [], [], [], []
    try:
        capture = NativeActionCapture(host, ActionCacheConfig())
        host.attach_method(capture)
        policy = ChunkPolicy(host, 8)
        metadata = dict(
            kind="independent_native_action_cache_population",
            backend=host.metadata(),
            sampler_parameters=asdict(host.parameters),
            source=source_metadata(),
            shard=shard,
            shards=shards,
            capture_schema=2,
            clean_action_source="native_final_ddim_sample",
            replay_population_sha256=replay_digest,
            sampling=(
                "exact frozen independent initialization replay"
                if frozen_states
                else "two fresh seeded placements per task, all 50 standard states excluded"
            ),
        )
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        started = time.monotonic()
        for replicate in range(2):
            for flat in range(40):
                if flat % shards != shard:
                    continue
                suite_id, task_id = divmod(flat, 10)
                suite_name = SUITES[suite_id]
                suite = benchmark.get_benchmark_dict()[suite_name]()
                task = suite.get_task(task_id)
                initial = np.asarray(suite.get_task_init_states(task_id))
                seed = 910000 + replicate * 1000 + flat
                random.seed(seed)
                np.random.seed(seed)
                env = OffScreenRenderEnv(
                    bddl_file_name=str(
                        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                    ),
                    camera_heights=256,
                    camera_widths=256,
                )
                try:
                    env.seed(seed)
                    env.reset()
                    state = np.asarray(env.sim.get_state().flatten()).copy()
                    if frozen_states:
                        state = frozen_states[(flat, replicate)].copy()
                    if initial.shape != (50, *state.shape) or any(
                        np.allclose(state[1:], old[1:], rtol=0, atol=1e-8) for old in initial
                    ):
                        raise RuntimeError("Independent state overlaps or cannot be verified")
                    context = EpisodeContext(suite_name, task_id, 1000 + replicate, seed)
                    np.savez_compressed(output / f"initial-{flat}-{replicate}.npz", state=state)
                    # Durable attempt accounting BEFORE rollout; an infrastructure failure
                    # cannot silently resample or increase the authorized episode count.
                    with (output / "attempts.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(
                                    suite=suite_name,
                                    task_id=task_id,
                                    episode_id=1000 + replicate,
                                    seed=seed,
                                    initial_state_sha256=stable_hash(state.tolist()),
                                )
                            )
                            + "\n"
                        )
                    result, _, _ = run_episode(env, policy, context, state, task.language, False)
                    result.update(
                        initial_state_sha256=stable_hash(state.tolist()),
                        excluded_standard_state_count=50,
                        pending_pairs=len(capture.pending),
                        retained_pairs=len(capture.pending) if result["success"] else 0,
                    )
                    if result["success"]:
                        for key, value, clean in capture.pending:
                            keys.append(key.numpy())
                            states.append(value.numpy())
                            clean_actions.append(clean.numpy())
                            partitions.append(suite_id)
                    records.append(result)
                    with (output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    print(json.dumps(result), flush=True)
                finally:
                    env.close()
        np.savez_compressed(
            output / "bank.npz",
            keys=np.asarray(keys, dtype=np.float32),
            states=np.asarray(states, dtype=np.float32),
            clean_actions=np.asarray(clean_actions, dtype=np.float32),
            suites=np.asarray(partitions, dtype=np.int64),
        )
        (output / "summary.json").write_text(
            json.dumps(
                dict(
                    episodes=len(records),
                    successes=sum(r["success"] for r in records),
                    pairs=len(keys),
                    wall_seconds=time.monotonic() - started,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--replay-population", type=Path, nargs="+")
    args = parser.parse_args()
    collect(
        args.model_path,
        args.libero_root,
        args.output_dir,
        args.shard,
        args.shards,
        args.replay_population,
    )


if __name__ == "__main__":
    main()
