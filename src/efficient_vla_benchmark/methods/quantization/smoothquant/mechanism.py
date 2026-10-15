"""Channel migration and signed W8 grids; no model or evaluation dependencies."""

import torch
from torch import nn

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
GROUPS = (
    ("input_layernorm", PROJECTIONS[:3]),
    ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
)


def validate_layer(layer):
    for name in PROJECTIONS:
        module = layer.get_submodule(name)
        if not isinstance(module, nn.Linear) or module.weight.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("SmoothQuant requires fresh native half-precision decoder linears")
    for norm_name, consumers in GROUPS:
        norm = layer.get_submodule(norm_name)
        if getattr(norm, "bias", None) is not None or norm.weight.ndim != 1:
            raise ValueError("SmoothQuant binding requires bias-free native RMSNorm")
        if any(layer.get_submodule(name).in_features != norm.weight.numel() for name in consumers):
            raise ValueError("Smoothing group geometry mismatch")


@torch.inference_mode()
def migration_scale(activation_max, weights, alpha):
    if activation_max.ndim != 1 or not bool(
        torch.isfinite(activation_max).all() & (activation_max >= 0).all()
    ):
        raise ValueError("Invalid activation channel maxima")
    maxima = (
        torch.stack([w.detach().float().abs().amax(0) for w in weights]).amax(0).clamp_min(1e-5)
    )
    if maxima.shape != activation_max.shape:
        raise ValueError("Smoothing channel geometry mismatch")
    return (activation_max.float().pow(alpha) / maxima.pow(1 - alpha)).clamp_min(1e-5)


@torch.inference_mode()
def quantize_weight(weight):
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Nonfinite smoothed weight")
    scales = weight.float().abs().amax(1).clamp_min(1e-5) / 127
    codes = (weight.float() / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return codes, scales


@torch.inference_mode()
def export_layer(layer, maxima, alpha):
    validate_layer(layer)
    norms, column_scales = {}, {}
    for norm_name, consumers in GROUPS:
        norm = layer.get_submodule(norm_name)
        weights = [layer.get_submodule(name).weight for name in consumers]
        scale = migration_scale(maxima[consumers[0]].to(norm.weight.device), weights, alpha)
        migrated_norm = (norm.weight.float() / scale).to(norm.weight.dtype)
        if not bool(torch.isfinite(migrated_norm).all()):
            raise ValueError("Smoothing overflow in native RMSNorm")
        norms[norm_name] = migrated_norm.cpu()
        for name in consumers:
            column_scales[name] = scale
    projections = {}
    for name in PROJECTIONS:
        module = layer.get_submodule(name)
        weight = module.weight.detach()
        if name in column_scales:
            weight = (weight.float() * column_scales[name]).to(weight.dtype)
        codes, scales = quantize_weight(weight)
        projections[name] = dict(
            codes=codes.cpu(),
            scales=scales.cpu(),
            native_dtype=str(module.weight.dtype),
            bias=None if module.bias is None else module.bias.detach().cpu(),
        )
    return dict(norms=norms, projections=projections)
