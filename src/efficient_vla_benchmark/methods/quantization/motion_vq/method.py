import gc

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash

from .artifact import read_manifest
from .packed import DualCodebookLinear
from .router import MotionRouter
from .target import replace, targets


class MotionVQ(InferenceMethod):
    name = "motion_vq"
    reversible = False
    requires_control_history = True

    def __init__(self, backend, config, root, prefix):
        m = read_manifest(root, config)
        if backend.method is not None or (m["backend"], m["checkpoint_sha256"]) != (
            backend.metadata()["backend"],
            backend.metadata()["checkpoint_sha256"],
        ):
            raise ValueError("MotionVQ native identity mismatch")
        native = targets(backend.model, prefix)
        entries = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        if set(entries) != set(native) or set(m["targets"]) != set(native):
            raise ValueError("MotionVQ coverage mismatch")
        prepared = {}
        for name, module in native.items():
            e = entries[name]
            for bank in ("high", "low"):
                if tuple(e[bank]["shape"]) != tuple(module.weight.shape) or e[bank][
                    "native_dtype"
                ] != str(module.weight.dtype):
                    raise ValueError("MotionVQ native geometry/dtype mismatch")
            if (e["bias"] is None) != (module.bias is None) or (
                module.bias is not None and not torch.equal(e["bias"], module.bias.detach().cpu())
            ):
                raise ValueError("MotionVQ native bias mismatch")
            prepared[name] = DualCodebookLinear(e)
        for name, op in prepared.items():
            native[name].cpu()
            replace(backend.model, name, op.to(backend.device))
        self.ops = list(prepared.values())
        self.router = MotionRouter(config.threshold)
        self.identity, self.config, self.backend = stable_hash(m), config, backend
        self.high_queries = self.low_queries = 0
        del prepared, entries, native
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self, context):
        self.router.reset()
        self.high_queries = self.low_queries = 0
        for op in self.ops:
            op.use_low = False

    def before_query(self, prepared):
        for op in self.ops:
            op.use_low = self.router.use_low
        if self.router.use_low:
            self.low_queries += 1
        else:
            self.high_queries += 1
        self.backend.query_diagnostics.update(self.control_diagnostics())

    def after_action(self, action, remaining):
        self.router.observe(action, remaining)
        return False

    def control_diagnostics(self):
        return dict(
            motion_vq_high_queries=self.high_queries,
            motion_vq_low_queries=self.low_queries,
            motion_vq_use_low=self.router.use_low,
        )

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            scope="224_LLM_projections_MotionVQ_core",
            variant="official_code_LLM_only_MotionVQ",
            runtime="packed_6_or_8bit_indices_tile_centroid_float_gemm",
            threshold=self.config.threshold,
            high=[256, 2, 256],
            low=[128, 2, 64],
            both_compressed_banks_resident=True,
            full_float_weight_cache=False,
            exclusions=(
                "ASIC spatial/temporal centroid reuse; "
                "vision/action quantization in public full-model export"
            ),
            routing="official >= boundary; previous executed H8 prefix final xyz magnitude",
            calibration=self.config.calibration_parameters(),
        )
