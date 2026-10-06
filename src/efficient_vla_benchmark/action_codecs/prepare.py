"""Extract raw actions only; all fitting and reconstruction use distinct train episodes."""

import argparse
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.calibration.libero import stratified_episode_indices
from efficient_vla_benchmark.evaluation.libero import SUITES

from .corpus import derive_prefix, digest, write_corpus


def extract(root, output, *, per_split=32, horizon=40, seed=814):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    if per_split < 1 or horizon < 1:
        raise ValueError("Positive split size and horizon required")
    tf.config.set_visible_devices([], "GPU")
    rows, datasets = [], {}
    for si, suite in enumerate(SUITES):
        directory = Path(root) / f"{suite}_no_noops" / "1.0.0"
        builder = tfds.builder_from_directory(str(directory))
        indices = stratified_episode_indices(
            builder.info.splits["train"].num_examples, per_split * 2, seed + si
        )
        rng = np.random.default_rng(seed + 1000 + si)
        assignment = dict(
            zip(
                rng.permutation(indices).tolist(),
                ["calibration"] * per_split + ["evaluation"] * per_split,
            )
        )
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.threading.max_intra_op_parallelism = 2
        ds = builder.as_dataset(
            split="train", shuffle_files=False, read_config=tfds.ReadConfig(options=options)
        )
        for ei, episode in enumerate(ds):
            if ei not in assignment:
                continue
            length = int(episode["steps"].cardinality().numpy())
            if length < horizon:
                raise ValueError(
                    "Selected episode shorter than common horizon; no silent replacement"
                )
            offset = int(rng.integers(0, length - horizon + 1))
            steps = list(episode["steps"].skip(offset).take(horizon).as_numpy_iterator())
            actions = np.stack([step["action"] for step in steps]).astype(np.float32)
            rows.append(
                {
                    "suite": suite,
                    "episode_index": ei,
                    "offset": offset,
                    "split": assignment.pop(ei),
                    "actions": actions,
                    "initial_state": steps[0]["observation"]["state"],
                }
            )
            if not assignment:
                break
        if assignment:
            raise ValueError("Dataset ended before all selected episodes")
        datasets[suite] = {
            "dataset_info_sha256": digest(directory / "dataset_info.json"),
            "split": "train",
            "version": "1.0.0",
        }
        print(f"Extracted {suite}: {per_split} calibration + {per_split} evaluation", flush=True)
    return write_corpus(
        output,
        rows,
        horizon=horizon,
        provenance={
            "datasets": datasets,
            "seed": seed,
            "per_split_per_suite": per_split,
            "selection": "seeded episode strata; disjoint random splits; one random window",
            "limit": "Demonstrations may already occur in released tokenizer pretraining",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset-root", type=Path)
    inputs.add_argument("--source-corpus", type=Path, help="Derive a fixed shorter prefix group")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-split", type=int, default=32)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--seed", type=int, default=814)
    a = p.parse_args()
    if a.source_corpus:
        derive_prefix(a.source_corpus, a.output, a.horizon)
    else:
        extract(a.dataset_root, a.output, per_split=a.per_split, horizon=a.horizon, seed=a.seed)


if __name__ == "__main__":
    main()
