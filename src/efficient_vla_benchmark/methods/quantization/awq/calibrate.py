"""Layerwise activation-aware scaling, clipping and real W4 packing; no training."""

import gc
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from .artifact import LINEARS, replace_linear
from .search import apply_layer_scales, batch_inputs, layer_arguments, search_layer_scales
from .vendor import reference_module


class CapturedEmbeddings(Exception):
    pass


@torch.inference_mode()
def capture_inputs(backend, dataset_root, config, layers):
    samples, identities = [], []

    def capture(module, args, kwargs):
        samples.append(args[0].detach().cpu())
        raise CapturedEmbeddings

    handle = layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for context, observation, identity in training_observations(
            dataset_root, config.calibration_samples, config.calibration_seed
        ):
            backend.reset(context)
            before = len(samples)
            try:
                backend.predict_chunk(observation)
            except CapturedEmbeddings:
                pass
            if len(samples) != before + 1:
                raise RuntimeError("Failed to capture exactly one multimodal decoder input")
            identities.append(identity)
            print(
                json.dumps({"stage": "capture", "sample": len(samples), "suite": context.suite}),
                flush=True,
            )
    finally:
        handle.remove()
    return samples, identities


@torch.inference_mode()
def calibrate(backend, config, dataset_root: Path, output: Path, target):
    if backend.method is not None:
        raise ValueError("AWQ calibration starts from an unmodified native backend")
    if target.precision != "bfloat16":
        raise ValueError("The first AWQ calibration recipe uses bfloat16")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibration output must be new or empty; incomplete runs are preserved")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.manual_seed(config.calibration_seed)
    quantize = reference_module("quantize.quantizer").pseudo_quantize_tensor
    clip = reference_module("quantize.auto_clip").auto_clip_layer
    WQLinear = reference_module("quantize.qmodule").WQLinear
    provenance = source_metadata()
    samples, identities = capture_inputs(backend, dataset_root, config, target.layers)
    if len(samples) != config.calibration_samples:
        raise ValueError("Calibration sample count is incomplete")
    target.offload()
    torch.cuda.empty_cache()
    states, valid = batch_inputs(samples, "cpu", target.dtype)
    del samples
    arguments = layer_arguments(valid.to(target.device), target.dtype, causal=target.causal)
    gpu_valid = valid.to(target.device)
    components, search_records = [], []
    q_config = {"zero_point": True, "q_group_size": config.group_size}
    layers = target.layers
    with torch.cuda.device(target.device):
        torch.cuda.reset_peak_memory_stats(target.device)
        for index, layer in enumerate(layers):
            layer_start = time.perf_counter()
            layer.to(target.device)
            features, handles = {}, []

            def recorder(name):
                def hook(module, inputs, result):
                    features[name] = inputs[0].detach().cpu()

                return hook

            for name in LINEARS:
                handles.append(layer.get_submodule(name).register_forward_hook(recorder(name)))
            try:
                next_states = layer(states.to(target.device), **arguments)[0].cpu()
            finally:
                for handle in handles:
                    handle.remove()
            scales, diagnostics = search_layer_scales(layer, features, gpu_valid, arguments, config)
            apply_layer_scales(layer, scales, features)
            # Reference AWQ deliberately leaves Q/K clipping disabled because
            # their errors interact through the attention dot product.
            for name in LINEARS:
                if name in {"self_attn.q_proj", "self_attn.k_proj"}:
                    continue
                linear = layer.get_submodule(name)
                inputs = features[name][valid].contiguous()
                token_samples = min(config.clip_sample_tokens, inputs.shape[0])
                maximum = clip(
                    linear.weight,
                    inputs,
                    n_bit=config.bits,
                    q_config=q_config,
                    n_grid=config.clip_grid,
                    max_shrink=config.clip_max_shrink,
                    n_sample_token=token_samples,
                ).to(device=linear.weight.device, dtype=linear.weight.dtype)
                grouped = linear.weight.reshape(*maximum.shape[:2], -1)
                grouped.copy_(torch.clamp(grouped, -maximum, maximum))
                del inputs, maximum
            features.clear()
            del scales
            for name in LINEARS:
                linear = layer.get_submodule(name)
                dequantized, weight_scales, zeros = quantize(
                    linear.weight, n_bit=config.bits, get_scale_zp=True, **q_config
                )
                linear.weight.copy_(dequantized)
                packed = WQLinear.from_linear(
                    linear, config.bits, config.group_size, False, weight_scales, zeros
                )
                replace_linear(layer, name, packed)
                del linear, dequantized, weight_scales, zeros, packed
            layer.cpu()
            filename = f"layer_{index:02d}.safetensors"
            save_file(
                {name: tensor.contiguous() for name, tensor in layer.state_dict().items()},
                str(output / filename),
            )
            components.append({"file": filename, "sha256": digest_file(output / filename)})
            states = next_states
            elapsed = time.perf_counter() - layer_start
            search_records.append({"layer": index, "seconds": elapsed, "scales": diagnostics})
            progress = {
                "complete": False,
                "layers_completed": index + 1,
                "elapsed_seconds": time.perf_counter() - start,
                "last_layer_seconds": elapsed,
            }
            (output / "progress.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps({"stage": "quantize", **progress}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize(target.device)
        peak = torch.cuda.max_memory_allocated(target.device)
    manifest = {
        "schema_version": 1,
        "algorithm": "awq",
        "complete": True,
        "backend": target.backend_id,
        "precision": target.precision,
        "scope": "language_decoder_block_linears",
        "config": config.calibration_parameters(),
        "checkpoint_sha256": target.checkpoint_sha256,
        "source": provenance,
        "calibration": identities,
        "components": components,
        "calibration_seconds": time.perf_counter() - start,
        "peak_layerwise_allocated_bytes": peak,
        "padding_excluded_from_scale_loss_and_clipping": True,
        "causal_attention": target.causal,
    }
    (output / "search.json").write_text(json.dumps(search_records, indent=2))
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(output / "manifest.json")
    (output / "progress.json").write_text(
        json.dumps(
            {
                "complete": True,
                "layers_completed": len(layers),
                "elapsed_seconds": manifest["calibration_seconds"],
            },
            indent=2,
        )
    )
    print(json.dumps({"stage": "complete", "seconds": manifest["calibration_seconds"]}), flush=True)
    return manifest
