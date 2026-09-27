"""Frozen, checksummed training-only task-level prominence distributions."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from .selection import instruction_key


def read_manifest(root, config):
    if root is None:
        raise ValueError("PACE requires a frozen training-demonstration calibration artifact")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "pace"
        or manifest.get("complete") is not True
        or manifest.get("split") != "train"
        or manifest.get("calibration_parameters") != config.calibration_parameters()
        or digest_file(root / "prominences.json") != manifest.get("statistics_sha256")
        or not manifest.get("samples")
        or any(item.get("split") != "train" for item in manifest["samples"])
        or stable_hash(manifest["samples"]) != manifest.get("samples_sha256")
    ):
        raise ValueError("PACE calibration is incompatible, incomplete or corrupted")
    data = json.loads((root / "prominences.json").read_text())
    if set(data) != set(manifest.get("instructions", [])) or len(data) != 40:
        raise ValueError("PACE joint-LIBERO calibration must cover all 40 training tasks")
    thresholds = {}
    counts = Counter(item.get("instruction") for item in manifest["samples"])
    indices = [item.get("episode_index") for item in manifest["samples"]]
    if (
        set(counts) != set(data)
        or type(manifest.get("episodes_per_task")) is not int
        or any(count != manifest["episodes_per_task"] for count in counts.values())
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("PACE calibration samples must uniquely cover every training task")
    for instruction, values in data.items():
        array = np.asarray(values, dtype=np.float64)
        if (
            instruction_key(instruction) != instruction
            or array.ndim != 1
            or not len(array)
            or not np.isfinite(array).all()
            or (array < 0).any()
        ):
            raise ValueError("Invalid PACE task calibration statistics")
        thresholds[instruction] = float(np.percentile(array, config.calibration_percentile))
    return manifest, thresholds
