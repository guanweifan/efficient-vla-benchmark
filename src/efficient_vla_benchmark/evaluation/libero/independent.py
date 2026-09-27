"""Independent frozen placements for authorized offline searches, not Dev/final."""

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash
from efficient_vla_benchmark.core.types import EpisodeContext

from .runner import MAX_STEPS, SUITES, WAIT_STEPS, configure_libero, run_episode


def validate_independent_state(state, standard, previous=()):
    state, standard = np.asarray(state), np.asarray(standard)
    if (
        state.ndim != 1
        or len(state) < 2
        or standard.shape != (50, *state.shape)
        or not np.isfinite(state).all()
        or not np.isfinite(standard).all()
    ):
        raise ValueError("Cannot verify independent placement geometry")
    # Simulator time is not a physical placement difference.
    if any(np.allclose(state[1:], old[1:], rtol=0, atol=1e-8) for old in [*standard, *previous]):
        raise ValueError("Independent placement overlaps standard or another search state")


def _task(suite_name, task_id):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    return task, suite, env, bddl


def create_population(libero_root, output, groups):
    """Freeze caller-declared groups before any outcome is observed; run no policy."""
    if output.exists() or not groups:
        raise ValueError("A new population destination and nonempty groups are required")
    contexts = [context for values in groups.values() for context in values]
    identities = [(c.suite, c.task_id, c.episode_id) for c in contexts]
    if len(set(identities)) != len(identities) or any(
        c.suite not in SUITES or not 0 <= c.task_id < 10 or c.episode_id < 50 for c in contexts
    ):
        raise ValueError("Invalid or duplicated independent episode identity")
    if any(not values for values in groups.values()):
        raise ValueError("Empty search groups are not allowed")
    configure_libero(libero_root, output / "_private/libero")
    records, previous = [], {}
    for group, values in groups.items():
        for context in values:
            random.seed(context.seed)
            np.random.seed(context.seed)
            task, suite, env, bddl = _task(context.suite, context.task_id)
            try:
                env.seed(context.seed)
                env.reset()
                state = np.asarray(env.sim.get_state().flatten()).copy()
                standard = np.asarray(suite.get_task_init_states(context.task_id))
                key = (context.suite, context.task_id)
                validate_independent_state(state, standard, previous.get(key, ()))
                previous.setdefault(key, []).append(state)
                filename = f"state-{len(records):04d}.npz"
                np.savez_compressed(output / filename, state=state)
                records.append(
                    dict(
                        group=group,
                        context=asdict(context),
                        filename=filename,
                        file_sha256=digest_file(output / filename),
                        state_sha256=stable_hash(state.tolist()),
                        standard_states_sha256=stable_hash(standard.tolist()),
                        bddl_sha256=digest_file(bddl),
                        excluded_standard_state_count=50,
                    )
                )
            finally:
                env.close()
    manifest = dict(schema_version=1, kind="independent_search_population", episodes=records)
    manifest["sha256"] = stable_hash(manifest)
    (output / "population.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_population(root):
    manifest = json.loads((root / "population.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "independent_search_population"
        or manifest.get("sha256")
        != stable_hash({k: v for k, v in manifest.items() if k != "sha256"})
        or not manifest.get("episodes")
    ):
        raise ValueError("Invalid independent population identity")
    seen, states = set(), {}
    for entry in manifest["episodes"]:
        context = EpisodeContext(**entry["context"])
        key = (context.suite, context.task_id, context.episode_id)
        filename = entry["filename"]
        if (
            key in seen
            or context.suite not in SUITES
            or not 0 <= context.task_id < 10
            or context.episode_id < 50
            or entry["excluded_standard_state_count"] != 50
            or Path(filename).name != filename
            or not filename.endswith(".npz")
            or digest_file(root / filename) != entry["file_sha256"]
        ):
            raise ValueError("Invalid/duplicated search state or modified state bytes")
        with np.load(root / filename, allow_pickle=False) as archive:
            state = np.asarray(archive["state"])
        if (
            state.ndim != 1
            or not np.isfinite(state).all()
            or stable_hash(state.tolist()) != entry["state_sha256"]
        ):
            raise ValueError("Search state content differs from frozen population")
        # This catches cross-group duplicates, irrespective of episode IDs/seeds.
        prior = states.setdefault(key[:2], [])
        if any(np.allclose(state[1:], old[1:], rtol=0, atol=1e-8) for old in prior):
            raise ValueError("Search groups share a physical placement")
        prior.append(state)
        seen.add(key)
    return manifest


def evaluate_population(policy, metadata, population_root, group, output, before_episode=None):
    """Use the exact shared H8 rollout loop; outcomes never update a population."""
    if output.exists():
        raise ValueError("Search run exists; never silently replace or retry its outcomes")
    manifest = load_population(population_root)
    selected = [e for e in manifest["episodes"] if e["group"] == group]
    if not selected:
        raise ValueError("Unknown independent population group")
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(
            dict(
                kind="independent_search_rollout",
                population_sha256=manifest["sha256"],
                group=group,
                episodes=selected,
                source=source_metadata(),
                policy=metadata,
                protocol=dict(wait_steps=WAIT_STEPS, max_steps=MAX_STEPS, environment_seed=0),
            ),
            indent=2,
        )
        + "\n"
    )
    records, started = [], time.monotonic()
    for entry in selected:
        context = EpisodeContext(**entry["context"])
        task, suite, env, bddl = _task(context.suite, context.task_id)
        try:
            with np.load(population_root / entry["filename"], allow_pickle=False) as archive:
                state = archive["state"]
            standard = np.asarray(suite.get_task_init_states(context.task_id))
            validate_independent_state(state, standard)
            if (
                stable_hash(standard.tolist()) != entry["standard_states_sha256"]
                or digest_file(bddl) != entry["bddl_sha256"]
            ):
                raise ValueError("Search simulator resources differ from the frozen population")
            if before_episode is not None:
                before_episode(context)
            try:
                result, _, _ = run_episode(env, policy, context, state, task.language, False)
            except Exception as error:
                result = dict(**asdict(context), status="error", success=False, error=repr(error))
                with (output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                raise
            result["initial_state_sha256"] = entry["state_sha256"]
            records.append(result)
            with (output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            env.close()
    summary = dict(
        kind="independent_search_rollout",
        population_sha256=manifest["sha256"],
        group=group,
        episodes=len(records),
        successes=sum(r["success"] for r in records),
        errors=0,
        wall_seconds=time.monotonic() - started,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
