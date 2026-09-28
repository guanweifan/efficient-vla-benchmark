"""Reject foreign or incomplete calibration instead of borrowing video coefficients."""

import json
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash


def read_manifest(root, config):
    if root is None:
        raise ValueError("TeaCache requires its independent training-calibration artifact")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    samples = manifest.get("samples", [])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "teacache"
        or manifest.get("backend") != "cogact"
        or manifest.get("signal") != "timestep_embedding"
        or manifest.get("complete") is not True
        or manifest.get("calibration_parameters") != config.calibration_parameters()
        or len(samples) != config.calibration_samples
        or any(sample.get("split") != "train" for sample in samples)
        or stable_hash(samples) != manifest.get("samples_sha256")
        or digest_file(root / "statistics.npz") != manifest.get("statistics_sha256")
    ):
        raise ValueError("TeaCache calibration is incomplete, corrupted or incompatible")
    with np.load(root / "statistics.npz", allow_pickle=False) as values:
        differences = values["input_changes"].copy()
        coefficients = values["coefficients"].copy()
        targets = values["residual_changes"].copy()
    steps = manifest["num_steps"]
    if (
        differences.shape != (steps,)
        or coefficients.shape != (config.polynomial_degree + 1,)
        or targets.shape != (config.calibration_samples, steps)
        or any(not np.isfinite(x).all() for x in (differences, coefficients, targets))
        or (differences < 0).any()
        or (targets < 0).any()
        or differences[0] != 0
        or (targets[:, 0] != 0).any()
        or manifest.get("cfg_rows") not in (1, 2)
        or len(manifest.get("timesteps", [])) != steps
        or any(type(t) is not int for t in manifest["timesteps"])
        or any(a <= b for a, b in zip(manifest["timesteps"], manifest["timesteps"][1:]))
    ):
        raise ValueError("Invalid TeaCache fitted statistics or noise grid")
    return manifest, differences, coefficients
