"""Fit relative-L1 residual changes using read-only balanced training observations."""

import json
import time
from dataclasses import asdict

import numpy as np
import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


def relative_changes(features):
    value = torch.stack(features).float().flatten(1)
    if value.shape[0] < 2 or not torch.isfinite(value).all():
        raise ValueError("Invalid calibration features")
    denominator = value[:-1].abs().mean(-1)
    if (denominator <= 0).any():
        raise ValueError("Zero norm cannot define relative L1 changes")
    changes = (value[1:] - value[:-1]).abs().mean(-1) / denominator
    return torch.cat((changes.new_zeros(1), changes)).cpu().numpy()


def fit_polynomial(input_changes, residual_changes, degree):
    x, y = (
        np.asarray(input_changes, dtype=np.float64),
        np.asarray(residual_changes, dtype=np.float64),
    )
    if x.ndim != 1 or y.ndim != 2 or y.shape[1] != len(x):
        raise ValueError("Incompatible input/output calibration observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y < 0).any():
        raise ValueError("Invalid calibration observations")
    if len(np.unique(x[1:])) <= degree:
        raise ValueError("Insufficient distinct timestep differences for polynomial fit")
    # Exclude the artificial first-step zero. Fit every training observation equally.
    coefficients = np.polyfit(np.tile(x[1:], y.shape[0]), y[:, 1:].reshape(-1), degree)
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite calibration coefficients")
    return coefficients


@torch.inference_mode()
def collect(backend, config, resources):
    if backend.method is not None:
        raise ValueError("TeaCache calibration requires an unmodified native policy")
    root = resources.artifact_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Calibration destination must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    net = backend.model_access.action_denoiser
    features, embeddings, times, identities, targets = [], [], [], [], []
    captured = {}
    patches = ForwardPatches()
    handles = []
    started = time.perf_counter()
    source = source_metadata()

    def first(module, args):
        captured["input"] = args[0].detach().clone()

    def last(module, args, output):
        features.append((output - captured.pop("input")).detach().clone())

    def embedding(module, args, output):
        embeddings.append(output.detach().clone())

    def wrapper(original):
        def forward(x, t, z):
            if not torch.equal(t, t[:1].expand_as(t)):
                raise ValueError("Calibration expects one noise time per native batch")
            times.append(int(t[0].item()))
            return original(x, t, z)

        return forward

    grid = differences = None
    rows = None
    try:
        handles.append(net.blocks[0].register_forward_pre_hook(first))
        handles.append(net.blocks[-1].register_forward_hook(last))
        handles.append(net.t_embedder.register_forward_hook(embedding))
        patches.wrap(net, wrapper)
        for context, observation, identity in training_observations(
            resources.dataset_path, config.calibration_samples, config.calibration_seed
        ):
            features.clear()
            embeddings.clear()
            times.clear()
            backend.reset(context)
            backend.predict_chunk(observation)
            if (
                len(features) != backend.parameters.num_ddim_steps
                or len(embeddings) != len(features)
                or captured
            ):
                raise RuntimeError("Incomplete native TeaCache capture")
            current = relative_changes(embeddings)
            if grid is not None and (times != grid or not np.array_equal(current, differences)):
                raise RuntimeError("Fixed timestep embedding or native schedule changed")
            grid, differences = list(times), current
            rows = features[0].shape[0]
            targets.append(relative_changes(features))
            identities.append(identity)
            print(
                json.dumps(dict(stage="teacache_calibration", samples=len(identities))), flush=True
            )
    finally:
        for handle in handles:
            handle.remove()
        patches.close()
        features.clear()
        embeddings.clear()
        captured.clear()
    if len(identities) != config.calibration_samples:
        raise RuntimeError("Incomplete balanced calibration population")
    targets = np.asarray(targets, dtype=np.float64)
    coefficients = fit_polynomial(differences, targets, config.polynomial_degree)
    prediction = np.polyval(coefficients, differences[1:])
    statistics = root / "statistics.npz"
    np.savez(
        statistics, input_changes=differences, residual_changes=targets, coefficients=coefficients
    )
    manifest = dict(
        schema_version=1,
        algorithm="teacache",
        backend="cogact",
        complete=True,
        signal="timestep_embedding",
        calibration_parameters=config.calibration_parameters(),
        samples=identities,
        samples_sha256=stable_hash(identities),
        checkpoint_sha256=backend._model_manifest["sha256"],
        sampler_parameters=asdict(backend.parameters),
        precision=str(next(net.parameters()).dtype),
        num_blocks=len(net.blocks),
        num_steps=backend.parameters.num_ddim_steps,
        cfg_rows=rows,
        timesteps=grid,
        statistics_sha256=digest_file(statistics),
        source=source,
        seconds=time.perf_counter() - started,
        fit_rmse=float(np.sqrt(np.mean((targets[:, 1:] - prediction) ** 2))),
        negative_fitted_increments=int((prediction < 0).sum()),
        scope="relative L1 timestep embeddings to full transformer-stack residual changes",
        calibration_scope="balanced four-suite training frames; no labels or gradients",
    )
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "manifest.json")
    return manifest
