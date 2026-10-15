"""Install a checked GPTQ artifact using one real packed W4 execution engine."""

import gc

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.quantization.packed import unpack_signed
from efficient_vla_benchmark.model_ops.quantization.tinychat import TinyChatLinear

from .artifact import read_manifest
from .mechanism import PROJECTIONS, validate_layer


class GPTQ(InferenceMethod):
    name = "gptq"
    reversible = False

    def __init__(self, backend, config, root, layers):
        manifest = read_manifest(root, config)
        native = backend.metadata()
        if (
            backend.method is not None
            or len(layers) != 32
            or manifest["backend"] != native["backend"]
            or manifest["checkpoint_sha256"] != native["checkpoint_sha256"]
        ):
            raise ValueError("GPTQ native backend/checkpoint mismatch")
        prepared = []
        # Prepare/validate every artifact on CPU before irreversible model mutation.
        for index, layer in enumerate(layers):
            validate_layer(layer)
            entry = torch.load(
                root / f"layer_{index:02d}.pt", map_location="cpu", weights_only=True
            )
            if set(entry) != set(PROJECTIONS):
                raise ValueError("GPTQ layer projection coverage mismatch")
            projections = {}
            for name, e in entry.items():
                module = layer.get_submodule(name)
                if (
                    e["shape"] != list(module.weight.shape)
                    or e["native_dtype"] != str(module.weight.dtype)
                    or (e["bias"] is None) != (module.bias is None)
                    or (
                        module.bias is not None
                        and not torch.equal(e["bias"], module.bias.detach().cpu())
                    )
                ):
                    raise ValueError("GPTQ artifact geometry/precision/bias mismatch")
                codes = unpack_signed(e["packed_codes"], 4, module.in_features)
                projections[name] = TinyChatLinear(codes, e["scales"], e["bias"])
            prepared.append(projections)
        for layer, projections in zip(layers, prepared):
            for name, replacement in projections.items():
                parent, attr = name.rsplit(".", 1)
                layer.get_submodule(name).cpu()
                setattr(layer.get_submodule(parent), attr, replacement.to(backend.device))
        self.identity, self.config = stable_hash(manifest), config
        del prepared, entry, projections
        gc.collect()
        torch.cuda.empty_cache()

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            calibration=self.config.calibration_parameters(),
            scope="decoder_w4g128_native_vision_and_action",
            targets=224,
            runtime="pinned_tinychat",
            weight_bits=4,
            activation_bits=16,
            scale_storage="float16",
            accumulation="float16",
            integer_mma=False,
            boundary_casts="native_bf16_to_fp16_and_back",
            calibration_reconstruction="native_bfloat16",
            full_dequantized_weight_cache=False,
        )
