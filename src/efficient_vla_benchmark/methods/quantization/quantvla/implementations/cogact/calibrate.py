"""Frozen training-only DuQuant/ATM/OHB calibration, using real packed student calls."""

import gc
import json
import time

import torch
from torch import nn

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from ...artifact import REVISION
from ...packed import PackedDuQuantLinear
from ...transforms import attention_scales, build_transform, quantize_mse, transformed_weight
from .attention import BalancedAttention
from .target import attentions, replace, targets


class CalibrationLinear(nn.Module):
    """Method-local offline observer; never installed by the runtime factory."""

    def __init__(self, op, config):
        super().__init__()
        self.op = op
        self.config = config
        self.calls = 0
        self.maximum = None

    def forward(self, inputs):
        op = self.op
        x = op.input_rotation(
            (inputs * op.smoothing.to(inputs.dtype)).index_select(-1, op.permutation)
        )
        if self.calls < self.config.calibration_samples:
            # Author PercentileCalibrator: maximum of per-call channel percentiles,
            # not a percentile across channel maxima from different calls.
            percentile = torch.quantile(
                x.detach().float().abs().reshape(-1, x.shape[-1]),
                self.config.activation_percentile / 100,
                dim=0,
            ).clamp_min(1e-6)
            self.maximum = (
                percentile if self.maximum is None else torch.maximum(self.maximum, percentile)
            )
            step = (self.maximum / 127).clamp_min(1e-6).to(x.dtype)
            op.linear.activation_step.copy_(step.float())
            self.calls += 1
        y = op.linear(x)
        if op.output_rotation is not None:
            y = op.output_rotation(y)
        return y if op.bias is None else y + op.bias.to(y.dtype)


@torch.inference_mode()
def collect_attention(backend, samples):
    originals = attentions(backend.model)
    values = {name: [] for name in originals}
    for name, module in originals.items():
        replace(
            backend.model,
            name,
            BalancedAttention(
                module, collector=lambda std, rms, name=name: values[name].append((std, rms))
            ),
        )
    try:
        for context, obs, _ in samples:
            backend.reset(context)
            backend.predict_chunk(obs)
    finally:
        for name, module in originals.items():
            replace(backend.model, name, module)
    if any(not v for v in values.values()):
        raise ValueError("Missing DiT attention calibration statistics")
    return {
        name: {
            "logits": torch.stack([x[0] for x in rows]).mean(0),
            "output": torch.stack([x[1] for x in rows]).mean(),
        }
        for name, rows in values.items()
    }


@torch.inference_mode()
def calibrate(backend, config, dataset, output):
    if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
        raise ValueError("Calibration requires fresh native BF16 CogACT")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite calibration evidence")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    native_metadata = backend.metadata()
    torch.manual_seed(config.calibration_seed)
    scale_samples = list(
        training_observations(dataset, config.calibration_samples, config.calibration_seed)
    )
    attention_samples = list(
        training_observations(
            dataset, config.attention_calibration_samples, config.calibration_seed
        )
    )
    print("QuantVLA: collecting native attention", flush=True)
    teacher = collect_attention(backend, attention_samples)
    modules = targets(backend.model)
    entries = {}
    wrappers = {}
    for index, (name, original) in enumerate(modules.items()):
        transform = build_transform(original.weight, config)
        weight = transformed_weight(original.weight, transform)
        codes, scales = quantize_mse(weight)
        serialized = {
            k: (
                [v.cpu() for v in value]
                if isinstance(value, list)
                else value.cpu()
                if torch.is_tensor(value)
                else value
            )
            for k, value in vars(transform).items()
        }
        entry = {
            "qweight": codes.cpu(),
            "weight_scale": scales.cpu(),
            "bias": None if original.bias is None else original.bias.detach().cpu(),
            "native_dtype": str(original.weight.dtype),
            "transform": serialized,
        }
        entries[name] = entry
        op = PackedDuQuantLinear(
            codes,
            scales,
            torch.ones(original.in_features, device=backend.device),
            original.bias,
            transform,
            frozen_step=True,
        ).to(backend.device)
        wrapper = CalibrationLinear(op, config)
        wrappers[name] = wrapper
        replace(backend.model, name, wrapper)
        original.cpu()
        del weight, codes, scales, transform, op
        if index % 16 == 0:
            print(f"QuantVLA: prepared {index + 1}/{len(modules)}", flush=True)
    del modules, original
    gc.collect()
    torch.cuda.empty_cache()
    print("QuantVLA: calibrating frozen activation scales", flush=True)
    for context, obs, _ in scale_samples:
        backend.reset(context)
        backend.predict_chunk(obs)
    for name, wrapper in wrappers.items():
        if wrapper.calls != config.calibration_samples:
            raise ValueError("Insufficient activation calibration calls")
        entries[name]["activation_step"] = wrapper.op.linear.activation_step.detach().cpu()
        replace(backend.model, name, wrapper.op)
    del wrappers
    print("QuantVLA: collecting packed-student attention", flush=True)
    student = collect_attention(backend, attention_samples)
    attention = {name: attention_scales(teacher[name], student[name], config) for name in teacher}
    torch.save(entries, output / "weights.pt")
    (output / "attention.json").write_text(json.dumps(attention, indent=2) + "\n")
    manifest = {
        "schema_version": 3,
        "algorithm": "quantvla",
        "complete": True,
        "backend": "cogact",
        "precision": "bfloat16",
        "checkpoint_sha256": native_metadata["checkpoint_sha256"],
        "config": config.calibration_parameters(),
        "reference_revision": REVISION,
        "target_modules": list(entries),
        "files": {n: digest_file(output / n) for n in ("weights.pt", "attention.json")},
        "calibration_identities": [s[2] for s in scale_samples],
        "attention_calibration_identities": [s[2] for s in attention_samples],
        "calibration_seconds": time.monotonic() - started,
        "source": source_metadata(),
        "calibration_changes": [
            "training_only_native_CogACT_inputs",
            "freeze_after_complete_author_PercentileCalibrator_budget_not_first_call",
            "packed_student_for_ATM_OHB_statistics",
        ],
        "execution": "packed_w4a8_tile_dequant_float_gemm_not_integer_mma",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
