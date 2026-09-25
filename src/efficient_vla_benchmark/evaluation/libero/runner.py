"""One LIBERO rollout for every policy, with explicit and resumable records."""

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from efficient_vla_benchmark.core.provenance import source_metadata, stable_hash
from efficient_vla_benchmark.core.types import EpisodeContext
from efficient_vla_benchmark.evaluation.traces import save_trace

from .manifest import episode_key, select_shard
from .observations import make_observation

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = dict(zip(SUITES, (220, 280, 300, 520)))
WAIT_STEPS = 10


def validate_protocol(suites, task_ids, trials, formal):
    """Reject invalid or accidentally large runs before allocating model or simulator."""
    if not suites or any(suite not in SUITES for suite in suites):
        raise ValueError("Unknown LIBERO suite")
    if len(set(suites)) != len(suites) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate suites or task indices are not allowed")
    if not task_ids or any(type(index) is not int or not 0 <= index < 10 for index in task_ids):
        raise ValueError("Each selected suite has task indices 0 through 9")
    if type(trials) is not int or not 1 <= trials <= 50:
        raise ValueError("Trials per task must be between 1 and 50")
    if not formal and len(suites) * len(task_ids) * trials > 4:
        raise ValueError("Development mode is limited to four episodes; use explicit formal mode")


def configure_libero(libero_root: Path, runtime_dir: Path, dataset_path: Path | None = None):
    root = libero_root.resolve() / "libero" / "libero"
    resources = {
        "benchmark_root": root,
        "assets": root / "assets",
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
    }
    if not all(path.is_dir() for path in resources.values()):
        raise ValueError("The supplied LIBERO source lacks required runtime resources")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    resources["datasets"] = dataset_path.resolve() if dataset_path else runtime_dir.resolve()
    (runtime_dir / "config.yaml").write_text(
        yaml.safe_dump({key: str(value) for key, value in resources.items()})
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_dir.resolve())
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def run_episode(env, policy, context, initial_state, instruction, record_trace=False):
    random.seed(context.seed)
    np.random.seed(context.seed)
    # Framework-specific random state is reset by the selected backend.
    policy.reset(context)
    env.seed(0)  # Match the fixed-initial-state official OFT evaluation.
    env.reset()
    raw = env.set_init_state(initial_state)
    for _ in range(WAIT_STEPS):
        raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    samples, sample_steps = [], []
    queries = 0
    for step in range(MAX_STEPS[context.suite]):
        depth_range = None
        if "agentview_depth" in raw:
            model = env.sim.model
            extent = model.stat.extent
            depth_range = (model.vis.map.znear * extent, model.vis.map.zfar * extent)
        obs = make_observation(raw, instruction, depth_range)
        decision = policy.step(obs)
        if decision.queried:
            queries += 1
        if record_trace:
            samples.append(obs)
            sample_steps.append(step)
        if decision.action.shape != (7,) or not np.isfinite(decision.action).all():
            raise ValueError("Policy returned an invalid environment action")
        raw, _, done, _ = env.step(decision.action.tolist())
        if done:
            return (
                {
                    **asdict(context),
                    "status": "completed",
                    "success": True,
                    "steps": step + 1,
                    "policy_queries": queries,
                },
                samples,
                sample_steps,
            )
    return (
        {
            **asdict(context),
            "status": "completed",
            "success": False,
            "steps": MAX_STEPS[context.suite],
            "policy_queries": queries,
        },
        samples,
        sample_steps,
    )


def evaluate(
    policy,
    backend,
    config,
    output_dir: Path,
    suites: list[str],
    task_ids: list[int],
    trials: int,
    seed: int,
    record_traces: bool,
    formal: bool,
    manifest=None,
    shard_index=0,
    num_shards=1,
):
    validate_protocol(suites, task_ids, trials, formal or manifest is not None)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    selected = (
        select_shard(manifest, shard_index, num_shards)
        if manifest
        else [
            dict(suite=s, task_id=t, episode_id=e, seed=seed)
            for s in suites
            for t in task_ids
            for e in range(trials)
        ]
    )
    expected = {episode_key(e): e for e in selected}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "kind": (manifest["kind"] if manifest else "formal" if formal else "development_smoke"),
        "config": asdict(config),
        "backend": backend.metadata(),
        "source": source_metadata(),
        "hardware": backend.runtime.hardware(),
        "protocol": {
            "suites": suites,
            "tasks": task_ids,
            "trials_per_task": trials,
            "seed": seed,
            "wait_steps": WAIT_STEPS,
            "max_steps": MAX_STEPS,
            "environment_seed": 0,
            "record_traces": record_traces,
            "depth": config.evaluation.get("depth", False),
            "trace_sampling": "all_control_steps",
        },
    }
    metadata["episode_manifest"] = (
        {
            "sha256": manifest["sha256"],
            "name": manifest["name"],
            "population": len(manifest["episodes"]),
        }
        if manifest
        else None
    )
    metadata["shard"] = {"index": shard_index, "count": num_shards, "episodes": selected}
    identity = stable_hash(
        {
            "config": metadata["config"],
            "backend": metadata["backend"],
            "protocol": metadata["protocol"],
            "runtime_source_sha256": metadata["source"]["runtime_source_sha256"],
            "runtime_packages": {
                name: value
                for name, value in metadata["source"]["packages"].items()
                if not name.startswith("efficient-vla-backend-")
                or name == "efficient-vla-backend-" + config.backend.replace("_", "-")
            },
            "cuda_extensions": metadata["source"]["cuda_extension_sha256"],
            "submodules": metadata["source"]["submodules"],
            "episode_manifest": metadata["episode_manifest"],
        }
    )
    metadata["experiment_id"] = identity
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing["experiment_id"] != identity or existing.get("shard") != metadata["shard"]:
            raise ValueError(
                "Output contains a different experiment; choose a new output directory"
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    results_path = output_dir / "episodes.jsonl"
    results = []
    if results_path.exists():
        results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    finished = {episode_key(r) for r in results}
    if len(finished) != len(results) or not finished <= expected.keys():
        raise ValueError("Existing results contain duplicates or foreign episodes")
    if any(r["seed"] != expected[episode_key(r)]["seed"] for r in results):
        raise ValueError("Existing result seed differs from manifest")
    for suite_name in suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in task_ids:
            task_episodes = [
                e for e in selected if e["suite"] == suite_name and e["task_id"] == task_id
            ]
            if not task_episodes:
                continue
            task = suite.get_task(task_id)
            if manifest and any(e["task_name"] != task.name for e in task_episodes):
                raise ValueError("Runtime task mapping differs from episode manifest")
            initial_states = suite.get_task_init_states(task_id)
            env = None
            try:
                for entry in task_episodes:
                    episode_id = entry["episode_id"]
                    key = (suite_name, task_id, episode_id)
                    if key in finished:
                        continue
                    context = EpisodeContext(suite_name, task_id, episode_id, entry["seed"])
                    start = time.monotonic()
                    try:
                        if env is None:
                            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder
                            env = OffScreenRenderEnv(
                                bddl_file_name=str(bddl / task.bddl_file),
                                camera_heights=256,
                                camera_widths=256,
                                camera_depths=config.evaluation.get("depth", False),
                            )
                        result, samples, steps = run_episode(
                            env,
                            policy,
                            context,
                            initial_states[episode_id],
                            task.language,
                            record_traces,
                        )
                        if record_traces:
                            trace_dir = output_dir / "traces"
                            trace_dir.mkdir(exist_ok=True)
                            save_trace(
                                trace_dir / f"{suite_name}-{task_id}-{episode_id}.npz",
                                context,
                                samples,
                                steps,
                            )
                    except Exception as exc:
                        result = {
                            **asdict(context),
                            "status": "error",
                            "success": False,
                            "error_type": type(exc).__name__,
                        }
                        # Private stderr may contain a diagnostic; public records never
                        # copy exception text, which can contain external paths.
                        import traceback

                        traceback.print_exc()
                        if env is not None:
                            env.close()
                            env = None
                    result["wall_seconds"] = time.monotonic() - start
                    with results_path.open("a") as stream:
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    results.append(result)
                    finished.add(key)
                    print(json.dumps(result), flush=True)
            finally:
                if env is not None:
                    env.close()
    summary = {
        "kind": metadata["kind"],
        "episodes": len(results),
        "expected_episodes": len(selected),
        "successes": sum(bool(r["success"]) for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "success_rate": sum(bool(r["success"]) for r in results) / len(results),
        "validation_completed": bool(manifest)
        and len(results) == len(selected)
        and all(r["status"] == "completed" for r in results),
        "formal_campaign_completed": (
            (formal or bool(manifest and manifest["kind"] == "final_evaluation"))
            and len(results) == len(selected)
            and num_shards == 1
            and all(r["status"] == "completed" for r in results)
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise RuntimeError("Evaluation contains errors; inspect private diagnostics")
    return summary
