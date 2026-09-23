"""Non-pickled, path-free observation traces for sequential timing."""

import json
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.types import EpisodeContext, Observation


def save_trace(
    path: Path,
    context: EpisodeContext,
    samples: list[Observation],
    steps: list[int],
    sampling: str = "control",
):
    if not samples:
        return
    with_depth = samples[0].depth is not None
    if any((sample.depth is not None) != with_depth for sample in samples):
        raise ValueError("A trace cannot mix RGB-only and RGB-depth observations")
    extra = {"depth": np.stack([sample.depth for sample in samples])} if with_depth else {}
    np.savez_compressed(
        path,
        image=np.stack([s.image for s in samples]),
        wrist_image=np.stack([s.wrist_image for s in samples]),
        proprio=np.stack([s.proprio for s in samples]),
        instruction=np.asarray(samples[0].instruction),
        steps=np.asarray(steps, dtype=np.int64),
        context=np.asarray(json.dumps(context.__dict__, sort_keys=True)),
        schema_version=np.asarray(3 if with_depth else 2),
        sampling=np.asarray(sampling),
        **extra,
    )


def load_trace(
    path: Path,
    require_control_steps: bool = False,
) -> tuple[EpisodeContext, list[Observation], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version not in {1, 2, 3}:
            raise ValueError("Unsupported trace schema")
        sampling = str(data["sampling"]) if version >= 2 else "query"
        if sampling not in {"control", "query"}:
            raise ValueError("Unsupported trace sampling")
        if require_control_steps and sampling != "control":
            raise ValueError("Control replay requires all control frames, not sparse query traces")
        context = EpisodeContext(**json.loads(str(data["context"])))
        images, wrists, proprio = data["image"], data["wrist_image"], data["proprio"]
        depths = data["depth"] if version == 3 else [None] * len(images)
        if len(depths) != len(images):
            raise ValueError("Inconsistent depth trace length")
        steps = data["steps"].copy()
        if not (len(images) == len(wrists) == len(proprio) == len(steps)):
            raise ValueError("Inconsistent trace lengths")
        if len(steps) == 0 or np.any(np.diff(steps) <= 0):
            raise ValueError("Trace steps must be nonempty and strictly increasing")
        if sampling == "control" and (steps[0] != 0 or np.any(np.diff(steps) != 1)):
            raise ValueError("Control traces must contain every step starting at zero")
        samples = [
            Observation(
                a.copy(),
                b.copy(),
                c.copy(),
                str(data["instruction"]),
                None if d is None else d.copy(),
            )
            for a, b, c, d in zip(images, wrists, proprio, depths)
        ]
    for sample in samples:
        sample.validate()
    return context, samples, steps
