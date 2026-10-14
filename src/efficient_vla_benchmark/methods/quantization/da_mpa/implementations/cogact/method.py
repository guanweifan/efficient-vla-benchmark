import gc

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear

from ...artifact import read_manifest
from ...drift import retained_blocks
from .target import replace, targets


class DAMPA(InferenceMethod):
    name = "da_mpa"
    reversible = False

    def __init__(self, backend, config, root):
        m = read_manifest(root, config)
        if (
            backend.method is not None
            or m["checkpoint_sha256"] != backend.metadata()["checkpoint_sha256"]
        ):
            raise ValueError("DA-MPA requires its native CogACT checkpoint")
        selected = targets(backend.model)
        entries = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        if set(entries) != set(selected) or set(m["targets"]) != set(selected):
            raise ValueError("DA-MPA target coverage mismatch")
        expected = {name for name, (_, side, _) in selected.items() if side == "dit_mlp"}
        if set(m["sensitivity"]) != expected:
            raise ValueError("Incomplete drift sensitivity scores")
        blocks = {block for _, side, block in selected.values() if side == "dit_mlp"}
        scores = {
            block: sum(
                m["sensitivity"][name]
                for name, (_, side, b) in selected.items()
                if side == "dit_mlp" and b == block
            )
            for block in blocks
        }
        retained = retained_blocks(
            scores, blocks, config.retention_ratio, config.exclude_final_blocks
        )
        prepared = {}
        for name, (module, side, block) in selected.items():
            e = entries[name]
            if (
                e["side"] != side
                or e["block"] != block
                or e["codes"].shape != module.weight.shape
                or e["native_dtype"] != str(module.weight.dtype)
            ):
                raise ValueError("DA-MPA target shape/dtype mismatch")
            if (e["bias"] is None) != (module.bias is None) or (
                module.bias is not None and not torch.equal(e["bias"], module.bias.detach().cpu())
            ):
                raise ValueError("DA-MPA bias mismatch")
            if side == "dit_mlp" and block in retained:
                continue
            prepared[name] = PackedLinear(
                e["codes"],
                e["scales"],
                bits=4,
                bias=e["bias"],
                activation_amax=e["activation_amax"],
                activation_bits=8,
            )
        for name, op in prepared.items():
            selected[name][0].cpu()
            replace(backend.model, name, op.to(backend.device))
        self.identity, self.config, self.retained, self.count = (
            stable_hash(m),
            config,
            sorted(retained),
            len(prepared),
        )
        del entries, prepared, selected
        gc.collect()
        torch.cuda.empty_cache()

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            scope="DA_MPA_independent_drift_allocation_core",
            variant="DA_MPA_only_no_CSRC",
            retained_native_dit_blocks=self.retained,
            retention_ratio=self.config.retention_ratio,
            quantized_targets=self.count,
            runtime="packed_W4A8_tile_dequant_float_gemm",
            integer_mma=False,
            exclusions="CSRC and associated SVD preconditioning; not full DA-PTQ",
            retained_dit_precision="native_float32_not_silently_BF16",
            calibration=self.config.calibration_parameters(),
        )
