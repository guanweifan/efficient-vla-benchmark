"""Capture native per-step residual features on balanced training observations."""

import json
import time
from dataclasses import asdict

import numpy as np
import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash


@torch.inference_mode()
def collect(backend, config, resources):
    if backend.method is not None:
        raise ValueError("BAC calibration requires a native, unmodified policy")
    root = resources.artifact_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Calibration destination must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    net = backend.model_access.action_denoiser
    branches = [module for block in net.blocks for module in (block.attn, block.mlp)]
    features = [[] for _ in branches]
    handles, identities = [], []
    similarities, errors = [], []
    started = time.perf_counter()

    def capture(index):
        def hook(module, args, output):
            features[index].append(output.detach().float().flatten().clone())

        return hook

    try:
        handles = [module.register_forward_hook(capture(i)) for i, module in enumerate(branches)]
        for context, observation, identity in training_observations(
            resources.dataset_path, config.calibration_samples, config.calibration_seed
        ):
            for values in features:
                values.clear()
            backend.reset(context)
            backend.predict_chunk(observation)
            if any(len(values) != backend.parameters.num_ddim_steps for values in features):
                raise RuntimeError("Incomplete BAC native denoising feature capture")
            sample_similarity, sample_errors = [], []
            for values in features:
                x = torch.stack(values)
                if not torch.isfinite(x).all():
                    raise ValueError("Nonfinite native calibration activations")
                normalized = torch.nn.functional.normalize(x, dim=1)
                sample_similarity.append((normalized @ normalized.T).clamp(-1, 1).cpu().numpy())
                # Eq. 13: all ordered pairs, averaged over steps and feature size.
                # Feature sizes match between every native attention/MLP branch.
                pair_l1 = torch.cdist(x, x, p=1) / x.shape[1]
                sample_errors.append(float(pair_l1.mean()))
            similarities.append(np.stack(sample_similarity))
            errors.append(sample_errors)
            identities.append(identity)
            print(json.dumps({"stage": "bac_calibration", "samples": len(identities)}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        features.clear()
    statistics = root / "statistics.npz"
    np.savez(
        statistics,
        similarity=np.mean(similarities, axis=0),
        l1_error=np.asarray(errors, dtype=np.float64).mean(0),
    )
    manifest = dict(
        schema_version=1,
        algorithm="bac",
        backend="cogact",
        complete=True,
        calibration_parameters=config.calibration_parameters(),
        samples=identities,
        samples_sha256=stable_hash(identities),
        checkpoint_sha256=backend._model_manifest["sha256"],
        sampler_parameters=asdict(backend.parameters),
        precision=str(next(net.parameters()).dtype),
        num_blocks=len(branches),
        num_steps=backend.parameters.num_ddim_steps,
        statistics_sha256=digest_file(statistics),
        source=source_metadata(),
        seconds=time.perf_counter() - started,
        reduction="mean per-observation source-to-target cosine; all-pairs normalized L1",
        scope="one shared schedule across four suites; native attention/MLP residuals",
    )
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "manifest.json")
    return manifest
