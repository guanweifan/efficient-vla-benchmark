"""One canonical DyQ implementation; backend binding only supplies decoder ownership."""

import gc
from collections import Counter

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash

from .artifact import read_manifest
from .dispatcher import DyQDispatcher
from .runtime import DynamicW4Linear
from .target import replace, targets


class DyQ(InferenceMethod):
    name = "dyq_vla"
    reversible = False
    requires_control_history = True

    def __init__(self, backend, config, root, prefix):
        m = read_manifest(root, config)
        if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
            raise ValueError("DyQ needs a fresh native BF16 model")
        if (m["checkpoint_sha256"], m["backend"]) != (
            backend.metadata()["checkpoint_sha256"],
            backend.metadata()["backend"],
        ):
            raise ValueError("DyQ checkpoint/backend mismatch")
        self.dispatcher = DyQDispatcher(config, m["thresholds"])
        native = targets(backend.model, prefix)
        entries = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        if set(entries) != set(native) or set(m["targets"]) != set(native):
            raise ValueError("DyQ target coverage mismatch")
        prepared = {}
        for name, module in native.items():
            e = entries[name]
            if e["codes"].shape != module.weight.shape or e["native_dtype"] != str(
                module.weight.dtype
            ):
                raise ValueError("DyQ weight shape/dtype mismatch")
            if (e["bias"] is None) != (module.bias is None) or (
                module.bias is not None and not torch.equal(e["bias"], module.bias.detach().cpu())
            ):
                raise ValueError("DyQ bias mismatch")
            prepared[name] = DynamicW4Linear(e)
        for name, op in prepared.items():
            native[name].cpu()
            replace(backend.model, name, op.to(backend.device))
        self.targets = list(prepared.values())
        self.backend = backend
        self.config = config
        self.identity = stable_hash(m)
        self.thresholds = m["thresholds"]
        self.query_bits = Counter()
        self.current_bits = 16
        del entries, native, prepared
        gc.collect()
        torch.cuda.empty_cache()

    def _set_bits(self, bits):
        if bits != self.current_bits:
            for op in self.targets:
                op.set_active_bits(bits)
            self.current_bits = bits

    def reset(self, context):
        self.dispatcher.reset()
        self.query_bits.clear()
        self._set_bits(16)

    def before_query(self, prepared):
        bits = self.dispatcher.active_bits
        self._set_bits(bits)
        self.query_bits[bits] += 1
        self.backend.query_diagnostics.update(self.dispatcher.metadata())

    def after_action(self, action, remaining):
        self.dispatcher.observe(action)
        if remaining == 0:
            self.dispatcher.dispatch()
        return False

    def control_diagnostics(self):
        return {**self.dispatcher.metadata(), "activation_bit_queries": dict(self.query_bits)}

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            scope="language_decoder_224_projections",
            variant="paper_temporal_core_packed_W4_dynamic_activation",
            weight_bits=4,
            activation_bits=[2, 4, 8, 16],
            thresholds=self.thresholds,
            calibration=self.config.calibration_parameters(),
            runtime="packed_W4_tile_dequant_float_gemm",
            integer_mma=False,
            dispatch="executed_native_commands; H8_chunk_boundary; Algorithm1",
            exclusions="author CUTLASS integer-MMA/async zero-copy engine unavailable",
            threshold_estimator="first conditional-mean-error crossing in equal-count bins",
            requires_control_history=True,
        )
