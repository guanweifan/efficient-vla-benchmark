"""Irreversible decoder conversion; all environment/control behavior stays native."""

import gc

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.quantization.integer_linear import IntegerLinear

from .artifact import read_manifest
from .mechanism import GROUPS, PROJECTIONS, validate_layer


class SmoothQuant(InferenceMethod):
    name = "smoothquant"
    reversible = False

    def __init__(self, backend, config, root, layers):
        manifest = read_manifest(root, config)
        metadata = backend.metadata()
        if (
            backend.method is not None
            or len(layers) != 32
            or (
                manifest["backend"] != metadata["backend"]
                or manifest["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or manifest.get("numerical_contract")
                != "fp32_statistics_native_folded_weights_tokenwise_int8_v1"
            )
        ):
            raise ValueError("SmoothQuant native checkpoint/backend/contract mismatch")
        prepared = []
        # Validate and prepare everything on CPU before mutating the native model.
        for index, layer in enumerate(layers):
            validate_layer(layer)
            entry = torch.load(
                root / f"layer_{index:02d}.pt", map_location="cpu", weights_only=True
            )
            if (
                set(entry) != {"norms", "projections"}
                or set(entry["projections"]) != set(PROJECTIONS)
                or set(entry["norms"]) != {g[0] for g in GROUPS}
            ):
                raise ValueError("SmoothQuant layer target coverage mismatch")
            for name, weight in entry["norms"].items():
                native = layer.get_submodule(name).weight
                if (
                    weight.shape != native.shape
                    or weight.dtype != native.dtype
                    or not bool(torch.isfinite(weight).all())
                ):
                    raise ValueError("Invalid smoothed native normalization")
            linears = {}
            for name, e in entry["projections"].items():
                native = layer.get_submodule(name)
                if e["codes"].shape != native.weight.shape or e["native_dtype"] != str(
                    native.weight.dtype
                ):
                    raise ValueError("SmoothQuant layer geometry/precision mismatch")
                if (e["bias"] is None) != (native.bias is None) or (
                    native.bias is not None
                    and not torch.equal(e["bias"], native.bias.detach().cpu())
                ):
                    raise ValueError("SmoothQuant native bias mismatch")
                linears[name] = IntegerLinear(e["codes"], e["scales"], bias=e["bias"])
            prepared.append((entry["norms"], linears))
        with torch.no_grad():
            for layer, (norms, linears) in zip(layers, prepared):
                for name, weight in norms.items():
                    layer.get_submodule(name).weight.copy_(weight)
                for name, replacement in linears.items():
                    parent, attr = name.rsplit(".", 1)
                    layer.get_submodule(name).cpu()
                    setattr(layer.get_submodule(parent), attr, replacement.to(backend.device))
        self.identity = stable_hash(manifest)
        self.config = config
        del prepared, entry, linears
        gc.collect()
        torch.cuda.empty_cache()

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            calibration=self.config.calibration_parameters(),
            scope="decoder_linear_w8a8_float_sdpa_native_vision_and_action",
            targets=224,
            smoothing="shared_qkv_and_gate_up_columns_folded_into_native_rmsnorm",
            runtime="triton_int8_mma_fp32_scaled_epilogue",
            integer_mma=True,
            weight_bits=8,
            activation_bits=8,
            activation_scale_axis="token_dynamic",
            accumulation="int32",
            full_dequantized_weight_cache=False,
        )
