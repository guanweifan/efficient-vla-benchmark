"""Balanced, reproducible observations from the four-suite RLDS training split."""

import hashlib
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.core.types import EpisodeContext, Observation
from efficient_vla_benchmark.evaluation.libero import SUITES


def stratified_episode_indices(episodes: int, count: int, seed: int) -> list[int]:
    if type(count) is not int or count < 1 or count > episodes:
        raise ValueError("Calibration needs distinct training episodes")
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, episodes, count + 1, dtype=np.int64)
    return [int(rng.integers(low, high)) for low, high in zip(edges[:-1], edges[1:])]


def observation_digest(observation: Observation) -> str:
    digest = hashlib.sha256()
    for value in (observation.image, observation.wrist_image, observation.proprio):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    digest.update(observation.instruction.encode())
    return digest.hexdigest()


def training_observations(root: Path, count: int = 32, seed: int = 0):
    for context, observation, identity, _ in _training_samples(root, count, seed, 1):
        yield context, observation, identity


def training_action_chunks(root: Path, count: int = 200, seed: int = 0, horizon: int = 8):
    """Distinct training episodes with complete, unpadded demonstration chunks.

    The seeded step is drawn from valid chunk starts, not selected by policy outcome.
    Actions remain raw dataset values; model-specific normalization belongs to bindings.
    """
    yield from _training_samples(root, count, seed, horizon)


def _training_samples(root: Path, count: int, seed: int, horizon: int):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    if type(count) is not int or count < 4 or count % 4:
        raise ValueError("Calibration sample count must be a positive multiple of four")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Action chunk horizon must be a positive integer")
    tf.config.set_visible_devices([], "GPU")
    for suite in SUITES:
        if not (root / f"{suite}_no_noops" / "1.0.0" / "dataset_info.json").is_file():
            raise ValueError("Calibration requires all four modified LIBERO RLDS training suites")
    for suite_index, suite in enumerate(SUITES):
        directory = root / f"{suite}_no_noops" / "1.0.0"
        builder = tfds.builder_from_directory(str(directory))
        total_episodes = builder.info.splits["train"].num_examples
        selected = stratified_episode_indices(total_episodes, count // 4, seed + suite_index)
        requested = set(selected)
        rng = np.random.default_rng(seed + 1000 + suite_index)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.threading.max_intra_op_parallelism = 2
        dataset = builder.as_dataset(
            split="train",
            shuffle_files=False,
            read_config=tfds.ReadConfig(options=options, shuffle_seed=seed),
        )
        for episode_index, episode in enumerate(dataset):
            if episode_index not in requested:
                continue
            length = int(episode["steps"].cardinality().numpy())
            if length < horizon:
                raise ValueError("Selected RLDS episode cannot supply a complete chunk")
            step_index = int(rng.integers(0, length - horizon + 1))
            steps = list(episode["steps"].skip(step_index).take(horizon))
            if len(steps) != horizon:
                raise ValueError("RLDS chunk length differs from its declared cardinality")
            step = steps[0]
            raw = step["observation"]
            # RLDS conversion already rotates simulator RGB by 180 degrees.
            # Do not apply make_observation's simulator-only rotation again.
            observation = Observation(
                raw["image"].numpy(),
                raw["wrist_image"].numpy(),
                raw["state"].numpy(),
                step["language_instruction"].numpy().decode(),
            )
            observation.validate()
            context = EpisodeContext(suite, -1, episode_index, seed)
            identity = {
                "suite": suite,
                "split": "train",
                "episode_index": episode_index,
                "step_index": step_index,
                "episode_steps": length,
                "dataset_info_sha256": digest_file(directory / "dataset_info.json"),
                "observation_sha256": observation_digest(observation),
            }
            actions = None
            if horizon > 1:
                actions = np.stack([s["action"].numpy() for s in steps])
                if actions.ndim != 2 or not np.isfinite(actions).all():
                    raise ValueError("Training action chunks must be finite two-dimensional arrays")
                identity["action_horizon"] = horizon
                identity["action_chunk_sha256"] = hashlib.sha256(
                    str((actions.shape, str(actions.dtype))).encode()
                    + np.ascontiguousarray(actions).tobytes()
                ).hexdigest()
            yield context, observation, identity, actions
            requested.remove(episode_index)
            if not requested:
                break
        if requested:
            raise ValueError("The RLDS dataset ended before selected episodes were reached")
