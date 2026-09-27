"""Read-only LeRobot-v2 demonstration calibration; no model loading or training."""

import argparse
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash

from .config import PACEConfig
from .selection import instruction_key, valleys


def calibrate(dataset, output, config, episodes_per_task=10, seed=0):
    import pyarrow.parquet as parquet

    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing calibration artifact")
    info = json.loads((dataset / "meta/info.json").read_text())
    train = info.get("splits", {}).get("train", "").split(":")
    if (
        info.get("codebase_version") != "v2.0"
        or info.get("total_tasks") != 40
        or len(train) != 2
        or type(episodes_per_task) is not int
        or episodes_per_task < 1
    ):
        raise ValueError("Expected a 40-task LeRobot-v2 LIBERO training export")
    start, end = map(int, train)
    tasks = {
        int(row["task_index"]): instruction_key(row["task"])
        for row in map(json.loads, (dataset / "meta/tasks.jsonl").read_text().splitlines())
    }
    if len(tasks) != 40 or len(set(tasks.values())) != 40:
        raise ValueError("Training task instructions must map uniquely to forty tasks")
    candidates = defaultdict(list)
    for row in map(json.loads, (dataset / "meta/episodes.jsonl").read_text().splitlines()):
        index = row["episode_index"]
        if start <= index < end and len(row["tasks"]) == 1:
            key = instruction_key(row["tasks"][0])
            if key not in tasks.values():
                raise ValueError("Unknown training task")
            candidates[key].append(index)
    if set(candidates) != set(tasks.values()):
        raise ValueError("Training episode metadata lacks task coverage")
    generator = np.random.default_rng(seed)
    distributions, samples = {}, []
    for task, key in sorted(tasks.items()):
        indices = sorted(candidates[key])
        if len(indices) < episodes_per_task:
            raise ValueError("Insufficient training demonstrations for a task")
        chosen = sorted(generator.choice(indices, episodes_per_task, replace=False).tolist())
        values = []
        for index in chosen:
            relative = info["data_path"].format(
                episode_chunk=index // info["chunks_size"], episode_index=index
            )
            path = (dataset / relative).resolve()
            if not path.is_relative_to(dataset.resolve()):
                raise ValueError("Dataset index escaped the selected dataset")
            table = parquet.read_table(
                path, columns=["actions", "task_index", "episode_index", "frame_index"]
            )
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            if (
                set(table["task_index"].to_pylist()) != {task}
                or set(table["episode_index"].to_pylist()) != {index}
                or table["frame_index"].to_pylist() != list(range(len(actions)))
            ):
                raise ValueError("Trajectory identity/order differs from training metadata")
            _, prominence = valleys(actions, config)
            values.extend(prominence.tolist())
            samples.append(
                dict(
                    split="train",
                    episode_index=index,
                    task_index=task,
                    instruction=key,
                    frames=len(actions),
                    actions_sha256=stable_hash(actions.tolist()),
                )
            )
        if not values:
            raise ValueError("Task has no training valleys; cannot invent a threshold")
        distributions[key] = sorted(values)
    output.mkdir(parents=True)
    statistics = output / "prominences.json"
    statistics.write_text(json.dumps(distributions, sort_keys=True, indent=2) + "\n")
    manifest = dict(
        schema_version=1,
        algorithm="pace",
        complete=True,
        split="train",
        dataset_format="LeRobot-v2 LIBERO Cartesian displacement commands",
        dataset_metadata_sha256=stable_hash(
            {
                name: digest_file(dataset / "meta" / name)
                for name in ("info.json", "tasks.jsonl", "episodes.jsonl")
            }
        ),
        instructions=sorted(distributions),
        calibration_parameters=config.calibration_parameters(),
        seed=seed,
        episodes_per_task=episodes_per_task,
        samples=samples,
        samples_sha256=stable_hash(samples),
        statistics_sha256=digest_file(statistics),
        source=source_metadata(),
        calibration_libraries={name: version(name) for name in ("numpy", "scipy", "pyarrow")},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--minimum-spacing", type=int, default=2)
    args = parser.parse_args()
    config = PACEConfig.parse(
        dict(smoothing_window=args.smoothing_window, minimum_spacing=args.minimum_spacing)
    )
    manifest = calibrate(args.dataset_path, args.output, config, args.episodes_per_task, args.seed)
    print(
        json.dumps(
            dict(
                tasks=len(manifest["instructions"]),
                episodes=len(manifest["samples"]),
                statistics_sha256=manifest["statistics_sha256"],
            )
        )
    )


if __name__ == "__main__":
    main()
