"""Training-only percentile provenance and native identity compatibility."""

import json
from pathlib import Path

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash


def native_identity(metadata):
    return {
        k: metadata[k]
        for k in (
            "backend",
            "model_variant",
            "checkpoint_sha256",
            "precision",
            "native_attention_implementation",
            "camera_views",
            "image_size",
        )
        if k in metadata
    }


def read_manifest(root, config, backend):
    if root is None:
        raise ValueError("TIES requires its backend-specific training quantiles")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    samples = manifest.get("samples", [])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "ties"
        or manifest.get("complete") is not True
        or manifest.get("backend") != backend
        or manifest.get("calibration_parameters") != config.calibration_parameters()
        or len(samples) != config.calibration_samples
        or any(s.get("split") != "train" for s in samples)
        or stable_hash(samples) != manifest.get("samples_sha256")
        or digest_file(root / "statistics.npz") != manifest.get("statistics_sha256")
    ):
        raise ValueError("TIES calibration is foreign, incomplete or corrupted")
    with np.load(root / "statistics.npz", allow_pickle=False) as data:
        taus = data["taus"]
    if (
        taus.shape != (config.calibration_samples,)
        or not np.isfinite(taus).all()
        or (np.abs(taus) > 1).any()
    ):
        raise ValueError("Invalid TIES training Kendall values")
    quantiles = np.quantile(taus, [0.25, 0.5, 0.75]).tolist()
    if quantiles != manifest.get("quantiles") or (
        config.mode == "soft" and quantiles[0] == quantiles[2]
    ):
        raise ValueError("Invalid or degenerate TIES calibration quantiles")
    return manifest
