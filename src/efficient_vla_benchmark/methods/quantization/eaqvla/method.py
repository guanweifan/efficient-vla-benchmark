"""Same modular policy on two explicitly bound architectures."""

import gc

import torch
from torch import nn

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear

from .artifact import read_manifest
from .packed import PackedPatchConv
from .target import replace, targets


class EaqVLA(InferenceMethod):
    name = "eaqvla"
    reversible = False

    def __init__(self, backend, config, root, vision_prefix, language_prefix):
        m = read_manifest(root, config)
        if (
            backend.method is not None
            or m["backend"] != backend.metadata()["backend"]
            or m["checkpoint_sha256"] != backend.metadata()["checkpoint_sha256"]
        ):
            raise ValueError("EaqVLA native checkpoint/backend mismatch")
        selected = targets(backend.model, vision_prefix, language_prefix)
        entries = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        if set(entries) != set(selected) or set(m["targets"]) != set(selected):
            raise ValueError("EaqVLA target coverage mismatch")
        prepared = {}
        for name, (module, side) in selected.items():
            e = entries[name]
            bits = config.vision_bits if side == "vision" else config.language_bits
            if (
                e["bits"] != bits
                or e["side"] != side
                or e["codes"].shape != module.weight.flatten(1).shape
                or e["native_dtype"] != str(module.weight.dtype)
            ):
                raise ValueError("EaqVLA target geometry/precision mismatch")
            if (e["bias"] is None) != (module.bias is None) or (
                module.bias is not None and not torch.equal(e["bias"], module.bias.detach().cpu())
            ):
                raise ValueError("Native bias mismatch")
            prepared[name] = (
                PackedPatchConv(module, e, bits)
                if isinstance(module, nn.Conv2d)
                else PackedLinear(e["codes"], e["scales"], bits=bits, bias=e["bias"])
            )
        for name, op in prepared.items():
            selected[name][0].cpu()
            replace(backend.model, name, op.to(backend.device))
        self.identity, self.config, self.count = stable_hash(m), config, len(selected)
        del entries, selected, prepared
        gc.collect()
        torch.cuda.empty_cache()

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            scope="vision_and_language_modular_GPTQ_projector_and_action_native",
            variant="modular_weight_GPTQ_native_activation",
            targets=self.count,
            vision_bits=self.config.vision_bits,
            language_bits=self.config.language_bits,
            runtime="packed_tile_dequant_float_gemm",
            integer_mma=False,
            activation_quantization=False,
            calibration=self.config.calibration_parameters(),
        )
