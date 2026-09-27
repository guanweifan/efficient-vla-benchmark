"""Checksummed native training statistics; never fit schedules to evaluation data."""

import json
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash


def read_manifest(root, config):
    if root is None:
        raise ValueError("BAC requires a frozen training-calibration artifact")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "bac"
        or manifest.get("backend") != "cogact"
        or manifest.get("complete") is not True
        or manifest.get("calibration_parameters") != config.calibration_parameters()
        or len(manifest.get("samples", [])) != config.calibration_samples
        or any(sample.get("split") != "train" for sample in manifest.get("samples", []))
        or digest_file(root / "statistics.npz") != manifest.get("statistics_sha256")
    ):
        raise ValueError("BAC calibration is incomplete, corrupted or incompatible")
    if stable_hash(manifest["samples"]) != manifest.get("samples_sha256"):
        raise ValueError("BAC sample provenance checksum mismatch")
    with np.load(root / "statistics.npz", allow_pickle=False) as stats:
        similarity, errors = stats["similarity"].copy(), stats["l1_error"].copy()
    n, t = manifest["num_blocks"], manifest["num_steps"]
    if (
        similarity.shape != (n, t, t)
        or errors.shape != (n,)
        or not np.isfinite(similarity).all()
        or not np.isfinite(errors).all()
        or (errors < 0).any()
        or np.abs(similarity).max() > 1.00001
        or not np.allclose(similarity, similarity.transpose(0, 2, 1), atol=1e-5)
    ):
        raise ValueError("Invalid BAC feature statistics")
    return manifest, similarity, errors
