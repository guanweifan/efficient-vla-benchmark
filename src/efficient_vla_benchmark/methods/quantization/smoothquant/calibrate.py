"""Collect native training statistics and export SmoothQuant without policy updates."""

import json
import time

import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from .mechanism import GROUPS, export_layer, validate_layer


@torch.inference_mode()
def calibrate(backend, config, resources, layers):
    output = resources.artifact_dir
    if backend.method is not None or output is None or output.exists() or len(layers) != 32:
        raise ValueError("SmoothQuant requires a fresh 32-layer model and new artifact directory")
    for layer in layers:
        validate_layer(layer)
    output.mkdir(parents=True)
    start = time.monotonic()
    maxima, calls, handles, identities = {}, {}, [], []

    def capture(key):
        def hook(module, args):
            values = args[0].detach().reshape(-1, module.in_features).float().abs().amax(0)
            maxima[key] = torch.maximum(maxima[key], values) if key in maxima else values
            calls[key] = calls.get(key, 0) + 1

        return hook

    try:
        for index, layer in enumerate(layers):
            for _, consumers in GROUPS:
                name = consumers[0]
                handles.append(
                    layer.get_submodule(name).register_forward_pre_hook(capture((index, name)))
                )
        for context, observation, identity in training_observations(
            resources.dataset_path, config.calibration_samples, config.calibration_seed
        ):
            previous = dict(calls)
            backend.reset(context)
            backend.predict_chunk(observation)
            if len(calls) != 64 or any(v != previous.get(k, 0) + 1 for k, v in calls.items()):
                raise ValueError("Every native decoder smoothing group must execute exactly once")
            identities.append(identity)
            print(
                json.dumps(dict(captured=len(identities), total=config.calibration_samples)),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(identities) != config.calibration_samples:
        raise ValueError("Incomplete SmoothQuant calibration")
    files = {}
    for index, layer in enumerate(layers):
        local = {name: maxima[(index, name)] for _, group in GROUPS for name in [group[0]]}
        entry = export_layer(layer, local, config.alpha)
        name = f"layer_{index:02d}.pt"
        torch.save(entry, output / name)
        files[name] = digest_file(output / name)
        print(json.dumps(dict(exported=index + 1, total=len(layers))), flush=True)
    metadata = backend.metadata()
    manifest = dict(
        schema_version=1,
        algorithm="smoothquant",
        complete=True,
        config=config.calibration_parameters(),
        layers=len(layers),
        files=files,
        backend=metadata["backend"],
        checkpoint_sha256=metadata["checkpoint_sha256"],
        calibration_identities=identities,
        source=source_metadata(),
        numerical_contract="fp32_statistics_native_folded_weights_tokenwise_int8_v1",
        wall_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
