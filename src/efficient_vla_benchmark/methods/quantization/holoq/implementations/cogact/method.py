"""Instance-owned timestep dispatch and irreversible packed conversion."""

import gc

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash

from ...artifact import read_manifest
from ...packed import HoloQLinear
from ...rotation import validate_transform
from .step import DDIMStepTracker, ddim_schedule
from .target import replace, targets


class HoloQ(InferenceMethod):
    name = "holoq"
    reversible = False

    def __init__(self, backend, config, root):
        self.tracker = None
        m = read_manifest(root, config)
        if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
            raise ValueError("HoloQ requires fresh native BF16 CogACT")
        if m["checkpoint_sha256"] != backend.metadata()["checkpoint_sha256"]:
            raise ValueError("HoloQ checkpoint mismatch")
        schedule = ddim_schedule(backend.model.action_model, backend.parameters.num_ddim_steps)
        if schedule != m["schedule"]:
            raise ValueError("HoloQ timestep/noise schedule mismatch")
        native = targets(backend.model)
        entries = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        if set(entries) != set(native) or set(m["targets"]) != set(native):
            raise ValueError("HoloQ artifact target coverage mismatch")
        prepared = {}
        for name, (module, side) in native.items():
            e = entries[name]
            if (
                e["codes"].shape != module.weight.shape
                or e["native_dtype"] != str(module.weight.dtype)
                or e["side"] != side
            ):
                raise ValueError("HoloQ weight geometry/dtype/side mismatch")
            validate_transform(
                e["transform"], module.in_features, module.out_features, side != "language"
            )
            expected_group = config.gptq_block_size if side == "language" else module.in_features
            if e["weight_group_size"] != expected_group:
                raise ValueError("HoloQ GPTQ scale grouping mismatch")
            rows = 1 if side == "language" else backend.parameters.num_ddim_steps
            scale = e["activation_steps"]
            if scale.shape != (rows, module.in_features) or not bool(
                torch.isfinite(scale).all() & (scale > 0).all()
            ):
                raise ValueError("Invalid HoloQ activation scale table")
            if (e["bias"] is None) != (module.bias is None) or (
                module.bias is not None and not torch.equal(e["bias"], module.bias.detach().cpu())
            ):
                raise ValueError("HoloQ native bias mismatch")
            prepared[name] = HoloQLinear(e, self.current_step)
        self.tracker = DDIMStepTracker(
            backend.model.action_model, backend.parameters.num_ddim_steps
        )
        try:
            for name, op in prepared.items():
                native[name][0].cpu()
                replace(backend.model, name, op.to(backend.device))
        except Exception:
            self.close()
            raise
        self.identity, self.config, self.schedule = stable_hash(m), config, schedule
        del entries, native, prepared
        gc.collect()
        torch.cuda.empty_cache()

    def current_step(self):
        return None if self.tracker is None else self.tracker.current_step

    def reset(self, context):
        if self.tracker is not None:
            self.tracker.reset()

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

    def metadata(self):
        return dict(
            artifact_sha256=self.identity,
            scope="224_language_and_48_dit_attention_mlp",
            variant="official_code_composite_gptq_perstep_cogact",
            runtime="packed_w4a4_tile_dequant_float_gemm",
            integer_mma=False,
            activation_storage="int8_codes_representing_signed_A4",
            activation_scale="input_channel; DiT additionally per_native_timestep",
            weight_scale="LLM_per_output_per_GPTQ_block; DiT_per_output",
            sampler=self.schedule,
            calibration=self.config.calibration_parameters(),
            paper_differences=(
                "official channelwise LLM A4 and DiT output rotation; not paper-v3 tokenwise LLM A4"
            ),
            backend_adaptation="native CogACT DDIM10/H8; no new sampler or backbone",
        )
