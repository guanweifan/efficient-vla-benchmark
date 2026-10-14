"""Packed channel allocation with explicit irreversible backend lifecycle."""

import gc
import json

import torch
from torch import nn

from efficient_vla_benchmark.core.method import InferenceMethod

from ...artifact import read_manifest, validate_gates
from ...packed import MixedBitConv2d, MixedBitLinear
from .target import replace, targets


class QVLA(InferenceMethod):
    name = "qvla"
    reversible = False

    def __init__(self, backend, config, root):
        manifest = read_manifest(root, config)
        if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
            raise ValueError("QVLA needs a fresh native BF16 OFT instance")
        if manifest["checkpoint_sha256"] != backend.metadata()["checkpoint_sha256"]:
            raise ValueError("QVLA checkpoint mismatch")
        native = targets(backend.model)
        gates = json.loads((root / "gates.json").read_text())
        self.identity = validate_gates(manifest, gates, native, config)
        prepared = {}
        for name, row in gates.items():
            module = native[name]
            prepared[name] = (
                MixedBitConv2d(module, torch.tensor(row))
                if isinstance(module, nn.Conv2d)
                else MixedBitLinear(module, torch.tensor(row))
            ).cpu()
        for name, module in prepared.items():
            # Validation/packing is complete. Drop the original GPU storage before
            # installation; the native dict otherwise keeps a second model resident.
            native[name].cpu()
            replace(backend.model, name, module.to(backend.device))
        self.config = config
        self.count = len(prepared)
        self.stats = manifest["stats"]
        self.histogram = {
            str(b): sum(row.count(b) for row in gates.values()) for b in config.candidate_bits
        }
        self.storage = sum(
            m.packed_weight.numel() * m.packed_weight.element_size()
            for m in backend.model.modules()
            if hasattr(m, "packed_weight")
        )
        del native, prepared
        gc.collect()
        torch.cuda.empty_cache()

    def metadata(self):
        return {
            "artifact_sha256": self.identity,
            "scope": "vision_and_language_active_linear_conv",
            "allocation_scope": "joint_zero_bit_and_quantization"
            if 0 in self.config.candidate_bits
            else "quantization_only_no_zero_bit",
            "calibration": self.config.calibration_parameters(),
            "packed_target_modules": self.count,
            "channel_counts_by_bit": self.histogram,
            "allocation": self.stats,
            "sensitivity": "official_input_Hessian_proxy_not_action_interventions",
            "runtime": "mixed_channel_packed_tile_dequant_float_gemm",
            "integer_mma": False,
            "packed_weight_bytes": self.storage,
            "activation_precision": "native_bfloat16",
            "weight_quantizer": "official_native_dtype_rounding_v2",
            "retained_precision_scope": "projectors_action_head_lm_head_and_16bit_rows",
            "budget_definition": "channel_average_not_parameter_weighted",
        }
