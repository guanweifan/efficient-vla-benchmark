"""AWQ scale search on padded multimodal batches, excluding padding from losses.

Use the reference's activation-power grid and groupwise quantizer. Unlike its
text-only driver, independent observations never attend across sample boundaries.
"""

import torch
from torch import nn

from .vendor import reference_module


def batch_inputs(samples, device, dtype):
    if not samples or any(sample.ndim != 3 or sample.shape[0] != 1 for sample in samples):
        raise ValueError("Calibration inputs must be nonempty batch-one decoder embeddings")
    length = max(sample.shape[1] for sample in samples)
    width = samples[0].shape[-1]
    inputs = torch.zeros((len(samples), length, width), dtype=dtype, device=device)
    valid = torch.zeros((len(samples), length), dtype=torch.bool, device=device)
    for index, sample in enumerate(samples):
        inputs[index, : sample.shape[1]] = sample[0].to(device=device, dtype=dtype)
        valid[index, : sample.shape[1]] = True
    return inputs, valid


def layer_arguments(valid, dtype):
    length = valid.shape[1]
    zeros = torch.zeros((), device=valid.device, dtype=dtype)
    masked = torch.full((), torch.finfo(dtype).min, device=valid.device, dtype=dtype)
    key_mask = torch.where(valid[:, None, None, :], zeros, masked)
    return {
        "attention_mask": key_mask.expand(-1, -1, length, -1),
        "position_ids": torch.arange(length, device=valid.device)
        .unsqueeze(0)
        .expand(valid.shape[0], -1),
        "cache_position": torch.arange(length, device=valid.device),
        "past_key_value": None,
        "output_attentions": False,
        "use_cache": False,
    }


@torch.inference_mode()
def search_scale(block, linears, inputs, valid, kwargs, config):
    quantize = reference_module("quantize.quantizer").pseudo_quantize_tensor
    originals = [linear.weight.detach().clone() for linear in linears]
    original_output = block(inputs, **kwargs)
    if isinstance(original_output, tuple):
        original_output = original_output[0]
    activation_scale = inputs[valid].abs().mean(dim=0)
    best_error, best_scale, best_ratio = float("inf"), None, None
    errors = []
    try:
        for index in range(config.scale_grid):
            ratio = index / config.scale_grid
            scale = activation_scale.pow(ratio).clamp(min=1e-4)
            scale = scale / (scale.max() * scale.min()).sqrt()
            for linear, original in zip(linears, originals):
                scaled = original * scale.view(1, -1)
                quantized = quantize(
                    scaled, n_bit=config.bits, zero_point=True, q_group_size=config.group_size
                )
                linear.weight.copy_(quantized / scale.view(1, -1))
            output = block(inputs, **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            loss = (original_output - output).float()[valid].square().mean().item()
            errors.append(loss)
            if loss < best_error:
                best_error, best_scale, best_ratio = loss, scale.detach().clone(), ratio
            for linear, original in zip(linears, originals):
                linear.weight.copy_(original)
    finally:
        for linear, original in zip(linears, originals):
            linear.weight.copy_(original)
    if best_scale is None or not torch.isfinite(best_scale).all():
        raise RuntimeError("AWQ scale search produced no finite candidate")
    return best_scale.cpu(), {"ratio": best_ratio, "mse": best_error, "candidate_mse": errors}


@torch.inference_mode()
def search_layer_scales(layer, features, valid, kwargs, config):
    groups = [
        (
            "input_layernorm",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            "self_attn",
            kwargs,
        ),
        ("self_attn.v_proj", ("self_attn.o_proj",), "self_attn.o_proj", {}),
        ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), "mlp", {}),
        ("mlp.up_proj", ("mlp.down_proj",), "mlp.down_proj", {}),
    ]
    results, statistics = [], {}
    device = next(layer.parameters()).device
    for previous, names, block_name, arguments in groups:
        block = layer.get_submodule(block_name)
        linears = [layer.get_submodule(name) for name in names]
        if (
            previous == "self_attn.v_proj"
            and layer.self_attn.v_proj.out_features != layer.self_attn.o_proj.in_features
        ):
            raise ValueError("OFT AWQ attention-output scaling requires matching V and O widths")
        inputs = features[names[0]].to(device)
        scale, diagnostic = search_scale(block, linears, inputs, valid, arguments, config)
        results.append((previous, names, scale))
        statistics[previous] = diagnostic
        del inputs
    return results, statistics


@torch.inference_mode()
def apply_layer_scales(layer, scales, features=None):
    for previous_name, names, cpu_scale in scales:
        previous = layer.get_submodule(previous_name)
        scale = cpu_scale.to(device=previous.weight.device, dtype=previous.weight.dtype)
        if isinstance(previous, nn.Linear):
            if previous.out_features != scale.numel():
                raise ValueError("Incompatible scale transfer between linear layers")
            previous.weight.div_(scale[:, None])
            if previous.bias is not None:
                previous.bias.div_(scale)
        else:
            previous.weight.div_(scale)
            if getattr(previous, "bias", None) is not None:
                previous.bias.div_(scale)
        for name in names:
            linear = layer.get_submodule(name)
            linear.weight.mul_(scale[None, :])
            if features is not None:
                features[name].div_(cpu_scale.to(features[name].dtype).view(1, 1, -1))
