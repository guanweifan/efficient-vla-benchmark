"""Portable, stratified development episodes; no policy or worker-specific sampling."""

import json
import random
from collections import Counter
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def episode_key(row):
    return row["suite"], row["task_id"], row["episode_id"]


def make_manifest(resources, sample_seed=20260924, policy_seed=0):
    """Select five of the first fifty official initial states for every task."""
    episodes = []
    for suite in SUITES:
        for task_id in range(10):
            task = resources[f"{suite}/{task_id}"]
            rng = random.Random(f"{sample_seed}:{suite}:{task['task_name']}")
            for index in sorted(rng.sample(range(50), 5)):
                episodes.append(
                    dict(
                        suite=suite,
                        task_id=task_id,
                        episode_id=index,
                        seed=policy_seed,
                        task_name=task["task_name"],
                    )
                )
    data = dict(
        schema_version=1,
        kind="development_validation",
        name="libero_dev200_v1",
        sample_seed=sample_seed,
        policy_seed=policy_seed,
        population_per_task=50,
        samples_per_task=5,
        task_order_index=0,
        resources=resources,
        episodes=episodes,
    )
    data["sha256"] = stable_hash(data)
    validate_manifest(data)
    return data


def validate_manifest(data):
    content = {k: v for k, v in data.items() if k != "sha256"}
    if data.get("sha256") != stable_hash(content):
        raise ValueError("Episode manifest digest mismatch")
    if (data.get("schema_version"), data.get("kind"), data.get("name")) != (
        1,
        "development_validation",
        "libero_dev200_v1",
    ):
        raise ValueError("Unsupported episode manifest")
    if (
        data.get("population_per_task"),
        data.get("samples_per_task"),
        data.get("task_order_index"),
    ) != (50, 5, 0):
        raise ValueError("Invalid Dev-200 protocol")
    episodes = data["episodes"]
    if len(episodes) != 200 or len({episode_key(e) for e in episodes}) != 200:
        raise ValueError("Dev-200 requires exactly 200 distinct episodes")
    counts = Counter((e["suite"], e["task_id"]) for e in episodes)
    if counts != Counter({(s, t): 5 for s in SUITES for t in range(10)}):
        raise ValueError("Dev-200 requires five episodes per task")
    for e in episodes:
        if (
            type(e["episode_id"]) is not int
            or not 0 <= e["episode_id"] < 50
            or type(e["task_id"]) is not int
            or type(e["seed"]) is not int
            or not 0 <= e["seed"] < 2**32
            or e["seed"] != data["policy_seed"]
        ):
            raise ValueError("Invalid episode index or seed")
        task = data["resources"][f"{e['suite']}/{e['task_id']}"]
        if e["task_name"] != task["task_name"]:
            raise ValueError("Task name differs from resource identity")
        for key in ("bddl", "init_states"):
            relative = Path(task[key]["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Resource paths must be portable and confined")
    return data


def load_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text()))


def select_shard(data, index=0, count=1):
    if type(count) is not int or type(index) is not int or not 1 <= count <= 200:
        raise ValueError("Shard count must be between 1 and 200")
    if not 0 <= index < count:
        raise ValueError("Shard index is outside shard count")
    # Keep a task's five initial states together where possible, amortizing environment setup.
    return [
        e for i, e in enumerate(data["episodes"]) if (i // 5 if count <= 40 else i) % count == index
    ]


def verify_resources(data, libero_root):
    root = Path(libero_root).resolve() / "libero" / "libero"
    for resource in data["resources"].values():
        for key in ("bddl", "init_states"):
            expected = resource[key]
            path = root / expected["path"]
            if not path.is_file() or digest_file(path) != expected["sha256"]:
                raise ValueError(f"LIBERO resource differs: {expected['path']}")


def collect_resources(libero_root):
    """Read the pinned official task mapping and initial states, never training episodes."""
    from libero.libero import benchmark

    root = Path(libero_root).resolve() / "libero" / "libero"
    resources = {}
    for name in SUITES:
        suite = benchmark.get_benchmark_dict()[name](task_order_index=0)
        if suite.n_tasks != 10:
            raise ValueError("Expected ten tasks per suite")
        for index in range(10):
            task = suite.get_task(index)
            if len(suite.get_task_init_states(index)) < 50:
                raise ValueError("Task has fewer than fifty official initial states")
            files = {
                "bddl": f"bddl_files/{task.problem_folder}/{task.bddl_file}",
                "init_states": f"init_files/{task.problem_folder}/{task.init_states_file}",
            }
            resources[f"{name}/{index}"] = {
                "task_name": task.name,
                **{k: {"path": v, "sha256": digest_file(root / v)} for k, v in files.items()},
            }
    return resources
