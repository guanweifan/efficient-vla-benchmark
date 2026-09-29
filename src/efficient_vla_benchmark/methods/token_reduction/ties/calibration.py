"""One native dense pass for each balanced training observation."""

import json
import time

import numpy as np
import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash

from .artifact import native_identity


@torch.inference_mode()
def collect(backend, config, resources, method_class):
    if backend.method is not None:
        raise ValueError("TIES calibration requires a native policy")
    root = resources.artifact_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Calibration destination must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    identity = native_identity(backend.metadata())
    method = method_class(backend, config, calibrating=True)
    values, samples = [], []
    started = time.perf_counter()
    try:
        backend.attach_method(method)
        for context, obs, sample in training_observations(
            resources.dataset_path, config.calibration_samples, config.calibration_seed
        ):
            backend.reset(context)
            backend.predict_chunk(obs)
            values.append(method.tau)
            samples.append(sample)
            print(json.dumps(dict(stage="ties_calibration", samples=len(samples))), flush=True)
    finally:
        backend.detach_method()
        method.close()
    if len(values) != config.calibration_samples:
        raise RuntimeError("Incomplete TIES calibration")
    taus = np.asarray(values, dtype=np.float64)
    if not np.isfinite(taus).all():
        raise ValueError("Undefined training Kendall values")
    quantiles = np.quantile(taus, [0.25, 0.5, 0.75]).tolist()
    if quantiles[0] == quantiles[2]:
        raise ValueError("Degenerate TIES training quantiles")
    path = root / "statistics.npz"
    np.savez(path, taus=taus)
    manifest = dict(
        schema_version=1,
        algorithm="ties",
        backend=identity["backend"],
        checkpoint_sha256=identity["checkpoint_sha256"],
        complete=True,
        native_identity=identity,
        calibration_parameters=config.calibration_parameters(),
        samples=samples,
        samples_sha256=stable_hash(samples),
        quantiles=quantiles,
        statistics_sha256=digest_file(path),
        seconds=time.perf_counter() - started,
        source=source_metadata(),
        calibration_scope="balanced four-suite train; no labels or gradients",
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
