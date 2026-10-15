"""Training-observation Hessian calibration; no gradients or policy optimization."""

import gc
import json
import math
import time

import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata
from efficient_vla_benchmark.model_ops.quantization.gptq import quantize
from efficient_vla_benchmark.model_ops.quantization.packed import pack_signed

from .mechanism import CONTRACT, GROUPS, capture_inputs, move_arguments, validate_layer


@torch.inference_mode()
def quantize_layer(layer, samples, config, device):
    """Global Hessians; later groups/layers observe already-quantized predecessors."""
    validate_layer(layer)
    artifact = {}
    for group in GROUPS:
        width = layer.get_submodule(group[0]).in_features
        hessian = torch.zeros((width, width), device=device, dtype=torch.float32)
        captured = [0]
        group_input = [None]

        def accumulate(module, args, covariance=hessian):
            x = args[0].detach()
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("GPTQ Hessian expects unpadded batch-one observations")
            previous = captured[0]
            captured[0] += 1
            covariance.mul_(previous / captured[0])
            values = x.reshape(-1, width).T.float() * math.sqrt(2 / captured[0])
            covariance.add_(values @ values.T)
            group_input[0] = x

        def check_shared(module, args):
            # Avoid assuming Q/K/V or gate/up see the same input on an unknown variant.
            if group_input[0] is None or not torch.equal(args[0], group_input[0]):
                raise ValueError("GPTQ shared-input group differs from native Llama semantics")

        # Q executes before K/V; gate executes before up in both native backends.
        first = (
            "self_attn.q_proj"
            if len(group) == 3
            else ("mlp.gate_proj" if len(group) == 2 else group[0])
        )
        handles = [layer.get_submodule(first).register_forward_pre_hook(accumulate)]
        handles += [
            layer.get_submodule(n).register_forward_pre_hook(check_shared)
            for n in group
            if n != first
        ]
        try:
            for hidden, arguments in samples:
                before = captured[0]
                layer(hidden.to(device), **move_arguments(arguments, device))
                if captured[0] != before + 1:
                    raise ValueError("Incomplete sequential GPTQ Hessian coverage")
                group_input[0] = None
        finally:
            for handle in handles:
                handle.remove()
        if captured[0] != config.calibration_samples:
            raise ValueError("Incomplete GPTQ calibration population")
        for name in group:
            module = layer.get_submodule(name)
            codes, scales = quantize(
                module.weight, hessian, 4, config.block_size, config.damp_percent, group_size=128
            )
            # Match the reference calibration driver: reconstruct in native model
            # precision for subsequent groups, not a reported runtime measurement.
            reconstructed = codes.float() * scales.repeat_interleave(128, 1)
            module.weight.copy_(reconstructed.to(module.weight.dtype))
            artifact[name] = dict(
                packed_codes=pack_signed(codes, 4).cpu(),
                scales=scales.cpu(),
                shape=list(codes.shape),
                native_dtype=str(module.weight.dtype),
                bias=None if module.bias is None else module.bias.detach().cpu(),
            )
            del codes, scales, reconstructed
        del hessian
    next_samples = [
        (layer(hidden.to(device), **move_arguments(arguments, device))[0].cpu(), arguments)
        for hidden, arguments in samples
    ]
    return artifact, next_samples


@torch.inference_mode()
def calibrate(backend, config, resources, layers, offload):
    output = resources.artifact_dir
    if backend.method is not None or output is None or output.exists() or len(layers) != 32:
        raise ValueError("GPTQ requires fresh 32-layer native model and new artifact directory")
    for layer in layers:
        validate_layer(layer)
    identity, source = backend.metadata(), source_metadata()
    output.mkdir(parents=True)
    start = time.monotonic()
    observations = training_observations(
        resources.dataset_path, config.calibration_samples, config.calibration_seed
    )
    samples, identities = capture_inputs(backend, observations, layers)
    if len(samples) != config.calibration_samples:
        raise ValueError("Incomplete GPTQ native embedding capture")
    offload()
    torch.cuda.empty_cache()
    files, layer_seconds = {}, []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for index, layer in enumerate(layers):
            before = time.monotonic()
            layer.to(backend.device)
            artifact, samples = quantize_layer(layer, samples, config, backend.device)
            layer.cpu()
            name = f"layer_{index:02d}.pt"
            torch.save(artifact, output / name)
            files[name] = digest_file(output / name)
            layer_seconds.append(time.monotonic() - before)
            print(
                json.dumps(
                    dict(layer=index, seconds=layer_seconds[-1], elapsed=time.monotonic() - start)
                ),
                flush=True,
            )
            del artifact
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    manifest = dict(
        schema_version=1,
        algorithm="gptq",
        complete=True,
        layers=len(layers),
        config=config.calibration_parameters(),
        files=files,
        backend=identity["backend"],
        checkpoint_sha256=identity["checkpoint_sha256"],
        calibration_identities=identities,
        source=source,
        numerical_contract=CONTRACT,
        layer_seconds=layer_seconds,
        wall_seconds=time.monotonic() - start,
    )
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(output / "manifest.json")
    return manifest
