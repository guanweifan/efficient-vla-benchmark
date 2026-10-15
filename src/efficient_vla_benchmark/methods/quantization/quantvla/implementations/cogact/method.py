"""Irreversible packed QuantVLA conversion with complete pre-mutation validation."""

import gc
import json
from types import SimpleNamespace

import torch

from efficient_vla_benchmark.core.method import InferenceMethod

from ...artifact import artifact_identity, read_manifest
from ...packed import PackedDuQuantLinear
from .attention import BalancedAttention
from .target import attentions, replace, targets


class QuantVLA(InferenceMethod):
    name = "quantvla"
    reversible = False

    def __init__(self, backend, config, root):
        manifest = read_manifest(root, config)
        if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
            raise ValueError("QuantVLA needs a fresh BF16 CogACT instance")
        if backend.metadata()["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError("QuantVLA checkpoint mismatch")
        native = targets(backend.model)
        attention = attentions(backend.model)
        entries = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        scales = json.loads((root / "attention.json").read_text())
        if (
            set(entries) != set(native)
            or set(manifest["target_modules"]) != set(native)
            or set(scales) != set(attention)
        ):
            raise ValueError("QuantVLA artifact target coverage mismatch")
        prepared = {}
        for name, m in native.items():
            e = entries[name]
            if e["qweight"].shape != m.weight.shape or e["native_dtype"] != str(m.weight.dtype):
                raise ValueError("QuantVLA weight geometry/precision mismatch")
            if (e["bias"] is None) != (m.bias is None) or (
                m.bias is not None and not torch.equal(e["bias"], m.bias.detach().cpu())
            ):
                raise ValueError("QuantVLA bias differs from the native checkpoint")
            prepared[name] = PackedDuQuantLinear(
                e["qweight"],
                e["weight_scale"],
                e["activation_step"],
                e["bias"],
                SimpleNamespace(**e["transform"]),
                frozen_step=True,
                runtime=config.runtime,
            )
            # Static rotations use each native layer's dtype. Do not recast
            # their full block stacks on every language/denoising forward.
            prepared[name].input_rotation.to(dtype=m.weight.dtype)
            if prepared[name].output_rotation is not None:
                prepared[name].output_rotation.to(dtype=m.weight.dtype)
        prepared_attention = {
            name: BalancedAttention(m, scales[name]) for name, m in attention.items()
        }
        for name, op in prepared.items():
            replace(backend.model, name, op.to(backend.device))
        for name, op in prepared_attention.items():
            replace(backend.model, name, op)
        self.identity = artifact_identity(manifest)
        self.config = config
        self.ops = tuple(prepared.values())
        del native, entries, prepared
        gc.collect()
        torch.cuda.empty_cache()

    def metadata(self):
        result = {
            "artifact_sha256": self.identity,
            "weight_bits": 4,
            "activation_bits": 8,
            "scope": "language_224_and_dit_mlp_24_with_native_fp32_attention_atm_ohb",
            "packed_linears": 248,
            "attention_modules": 12,
            "calibration": self.config.calibration_parameters(),
            "runtime": "packed_w4a8_tile_dequant_float_gemm",
            "integer_mma": False,
            "author_scope_adaptation": "CogACT native DDIM10 H8; offline frozen calibration",
            "activation_scale": "per_input_channel_frozen_step",
            "activation_integer_range": [-128, 127],
            "weight_scale": "author_five_point_mse_per_output_row",
            "calibration_observations": "training_only_balanced_four_suite",
            "no_full_float_weight_cache": True,
        }
        if self.config.runtime == "tinychat_fp16":
            result.update(
                runtime=self.config.execution(),
                quantized_gemm_precision="float16_products_and_accumulation",
                native_boundaries="bfloat16_language_float32_dit",
                activation_preparation="native_rounding_int8_codes_then_fp16",
                rotations="fused_blocks_native_short_row_bmm",
                precision_change_requires_new_development_evidence=True,
            )
        return result
