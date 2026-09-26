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


def collect(model_path, libero_root, output, shard, shards):
    if not 1 <= shards <= 40 or not 0 <= shard < shards:
        raise ValueError("Invalid independent population shard")
    if output.exists():
        raise ValueError("Population output exists; inspect it instead of replacing or resampling")
    configure_libero(libero_root, output / "_private/libero")
    from efficient_vla_backend_cogact.backend import CogACTBackend
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from .capture import NativeActionCapture

    host = CogACTBackend(model_path)
    keys, states, partitions, records = [], [], [], []
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
            sampling="two fresh seeded placements per task, all 50 standard states excluded",
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
                    if initial.shape != (50, *state.shape) or any(
                        np.allclose(state[1:], old[1:], rtol=0, atol=1e-8) for old in initial
                    ):
                        raise RuntimeError("Independent state overlaps or cannot be verified")
                    context = EpisodeContext(suite_name, task_id, 1000 + replicate, seed)
                    result, _, _ = run_episode(env, policy, context, state, task.language, False)
                    result.update(
                        initial_state_sha256=stable_hash(state.tolist()),
                        excluded_standard_state_count=50,
                        pending_pairs=len(capture.pending),
                        retained_pairs=len(capture.pending) if result["success"] else 0,
                    )
                    if result["success"]:
                        for key, value in capture.pending:
                            keys.append(key.numpy())
                            states.append(value.numpy())
                            partitions.append(suite_id)
                    records.append(result)
                    with (output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    np.savez_compressed(output / f"initial-{flat}-{replicate}.npz", state=state)
                    print(json.dumps(result), flush=True)
                finally:
                    env.close()
        np.savez_compressed(
            output / "bank.npz",
            keys=np.asarray(keys, dtype=np.float32),
            states=np.asarray(states, dtype=np.float32),
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
    args = parser.parse_args()
    collect(args.model_path, args.libero_root, args.output_dir, args.shard, args.shards)


if __name__ == "__main__":
    main()
