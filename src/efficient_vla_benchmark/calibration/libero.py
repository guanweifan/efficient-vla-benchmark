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
    import tensorflow as tf
    import tensorflow_datasets as tfds

    if type(count) is not int or count < 4 or count % 4:
        raise ValueError("Calibration sample count must be a positive multiple of four")
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
            if length <= 0:
                raise ValueError("Selected RLDS episode has unknown or empty cardinality")
            step_index = int(rng.integers(0, length))
            step = next(iter(episode["steps"].skip(step_index).take(1)))
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
            yield context, observation, identity
            requested.remove(episode_index)
            if not requested:
                break
        if requested:
            raise ValueError("The RLDS dataset ended before selected episodes were reached")
