"""Portable, stratified development episodes; no policy or worker-specific sampling."""

import json
import random
from collections import Counter
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PROTOCOLS = {
    "libero_dev200_v1": ("development_validation", 5),
    "libero_timing40_v1": ("timing_reference", 1),
    "libero_final2000_v1": ("final_evaluation", 50),
}


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
    protocol = PROTOCOLS.get(data.get("name"))
    if protocol is None or data.get("schema_version") != 1 or data.get("kind") != protocol[0]:
        raise ValueError("Unsupported episode manifest")
    samples = protocol[1]
    if (
        data.get("population_per_task"),
        data.get("samples_per_task"),
        data.get("task_order_index"),
    ) != (50, samples, 0):
        raise ValueError("Invalid fixed episode protocol")
    episodes = data["episodes"]
    population = 40 * samples
    if len(episodes) != population or len({episode_key(e) for e in episodes}) != population:
        raise ValueError(f"Manifest requires exactly {population} distinct episodes")
    counts = Counter((e["suite"], e["task_id"]) for e in episodes)
    if counts != Counter({(s, t): samples for s in SUITES for t in range(10)}):
        raise ValueError("Manifest requires its fixed episode count per task")
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
    if samples != 5:
        development = make_manifest(data["resources"], data["sample_seed"], data["policy_seed"])
        if data.get("development_manifest_sha256") != development["sha256"]:
            raise ValueError("Derived manifest does not match its fixed development source")
        if samples == 1 and episodes != development["episodes"][::5]:
            raise ValueError("Timing requires the lowest selected initial-state ID per task")
    return data


def derive_manifest(development, purpose):
    """Freeze workload/final identities without running a policy or selecting by outcomes."""
    validate_manifest(development)
    if development["name"] != "libero_dev200_v1" or purpose not in {"timing", "final"}:
        raise ValueError("Derivation requires Dev-200 and timing/final purpose")
    data = {k: v for k, v in development.items() if k != "sha256"}
    data["development_manifest_sha256"] = development["sha256"]
    data["name"] = "libero_timing40_v1" if purpose == "timing" else "libero_final2000_v1"
    data["kind"], data["samples_per_task"] = PROTOCOLS[data["name"]]
    data["episodes"] = (
        development["episodes"][::5]
        if purpose == "timing"
        else [
            dict(row, episode_id=index)
            for row in development["episodes"][::5]
            for index in range(50)
        ]
    )
    data["sha256"] = stable_hash(data)
    return validate_manifest(data)


def load_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text()))


def select_shard(data, index=0, count=1):
    if type(count) is not int or type(index) is not int or not 1 <= count <= len(data["episodes"]):
        raise ValueError("Shard count must be between 1 and the episode count")
    if not 0 <= index < count:
        raise ValueError("Shard index is outside shard count")
    # Keep a task's five initial states together where possible, amortizing environment setup.
    return [
        e
        for i, e in enumerate(data["episodes"])
        if (i // data["samples_per_task"] if count <= 40 else i) % count == index
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
