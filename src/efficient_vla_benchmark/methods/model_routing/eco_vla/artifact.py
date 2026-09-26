"""Read-only training-calibrated channel energies, separate from the checkpoint."""

import json
import time

import torch
from safetensors.torch import load_file, save_file

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash


def read_manifest(root, config, backend_id):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("EcoVLA requires a completed training-calibration artifact")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "eco_vla"
        or manifest.get("complete") is not True
        or manifest.get("backend") != backend_id
        or manifest.get("calibration_parameters") != config.calibration_parameters()
        or len(manifest.get("samples", [])) != config.calibration_samples
    ):
        raise ValueError("Incompatible or incomplete EcoVLA artifact")
    if not (root / "energy.safetensors").is_file() or digest_file(
        root / "energy.safetensors"
    ) != manifest.get("energy_sha256"):
        raise ValueError("EcoVLA calibration energy checksum failed")
    return manifest


def load_artifact(backend, decoder, root, config, backend_id):
    manifest = read_manifest(root, config, backend_id)
    if (
        manifest["checkpoint_sha256"] != backend._model_manifest["sha256"]
        or manifest["precision"] != str(next(decoder.parameters()).dtype)
        or manifest["num_layers"] != len(decoder.layers)
    ):
        raise ValueError("EcoVLA calibration belongs to another checkpoint or precision")
    energy = load_file(str(root / "energy.safetensors"))
    dims = {
        "mlp": decoder.layers[0].mlp.down_proj.in_features,
        "attention": decoder.layers[0].self_attn.o_proj.in_features,
    }
    for name, width in dims.items():
        if name not in energy or energy[name].shape != (len(decoder.layers), width):
            raise ValueError("EcoVLA energy geometry does not match the native decoder")
        if not torch.isfinite(energy[name]).all() or (energy[name] < 0).any():
            raise ValueError("EcoVLA energies must be finite and nonnegative")
    return energy, stable_hash(manifest)


@torch.inference_mode()
def calibrate(backend, decoder, config, resources, backend_id):
    if backend.method is not None:
        raise ValueError("EcoVLA calibration requires an unmodified native policy")
    output = resources.artifact_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibration output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    handles, current, samples, identities = [], {}, [], []

    def capture(name, index):
        def hook(module, args):
            key = name, index
            if key in current:
                raise RuntimeError("Expected one native projection per sample")
            current[key] = args[0].float().square().sum(dim=(0, 1))

        return hook

    try:
        for index, layer in enumerate(decoder.layers):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture("mlp", index)))
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(capture("attention", index))
            )
        for context, observation, identity in training_observations(
            resources.dataset_path, config.calibration_samples, config.calibration_seed
        ):
            current.clear()
            backend.reset(context)
            backend.predict_chunk(observation)
            if len(current) != 2 * len(decoder.layers):
                raise RuntimeError("Incomplete EcoVLA calibration sample")
            samples.append(
                {
                    name: torch.stack([current[name, i] for i in range(len(decoder.layers))]).cpu()
                    for name in ("mlp", "attention")
                }
            )
            identities.append(identity)
            print(json.dumps({"stage": "eco_calibration", "samples": len(identities)}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    energy = {
        name: torch.stack([sample[name] for sample in samples]).mean(0).contiguous()
        for name in ("mlp", "attention")
    }
    save_file(energy, str(output / "energy.safetensors"))
    manifest = {
        "schema_version": 1,
        "algorithm": "eco_vla",
        "complete": True,
        "backend": backend_id,
        "checkpoint_sha256": backend._model_manifest["sha256"],
        "precision": str(next(decoder.parameters()).dtype),
        "num_layers": len(decoder.layers),
        "energy_sha256": digest_file(output / "energy.safetensors"),
        "calibration_parameters": config.calibration_parameters(),
        "samples": identities,
        "source": source_metadata(),
        "seconds": time.perf_counter() - start,
        "reduction": "Eq. 8 sum over native tokens, averaged equally over training samples",
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(output / "manifest.json")
    return manifest
