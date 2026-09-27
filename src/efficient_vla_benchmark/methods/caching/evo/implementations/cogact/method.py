"""Native residual branches follow one fixed global layout per action query."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...search import validate_layout


class EVO(InferenceMethod):
    name = "evo"

    def __init__(self, backend, config, manifest):
        self.backend, self.config, self.manifest = backend, config, manifest
        self.net = backend.model_access.action_denoiser
        branches = [module for block in self.net.blocks for module in (block.attn, block.mlp)]
        if (
            self.net.training
            or manifest["checkpoint_sha256"] != backend._model_manifest["sha256"]
            or manifest["sampler_parameters"] != asdict(backend.parameters)
            or manifest["precision"] != str(next(self.net.parameters()).dtype)
            or manifest["num_blocks"] != len(branches)
            or manifest["num_steps"] != backend.parameters.num_ddim_steps
        ):
            raise ValueError("EVO schedule differs from checkpoint, sampler or branch geometry")
        self.selected = manifest["layouts"][str(config.refresh_budget)]
        self.layout = validate_layout(
            self.selected["layout"], len(branches), manifest["num_steps"], config.refresh_budget
        )
        self.updates = [frozenset(row) for row in self.layout]
        self.patches, self.closed = ForwardPatches(), False
        self.reset(None)
        if config.enabled:
            if hasattr(self.net, "_evb_evo"):
                raise ValueError("EVO already attached")
            self.net._evb_evo = self
            self.patches.wrap(self.net, self._denoiser)
            for index, branch in enumerate(branches):
                self.patches.wrap(branch, lambda original, i=index: self._branch(original, i))

    def _denoiser(self, original):
        def forward(x, t, z):
            if (
                not self.query_open
                or self.in_forward
                or torch.is_grad_enabled()
                or self.net.training
            ):
                raise RuntimeError("EVO requires a nonnested native inference query")
            self.step += 1
            if self.step >= self.manifest["num_steps"]:
                raise RuntimeError("EVO sampler exceeded the frozen lattice")
            self.in_forward = True
            try:
                return original(x, t, z)
            except Exception:
                self.reset(None)
                raise
            finally:
                self.in_forward = False

        return forward

    def _branch(self, original, index):
        def forward(x):
            if not self.in_forward:
                raise RuntimeError("EVO branch called outside native denoising")
            if self.step in self.updates[index]:
                output = original(x)
                self.saved[index] = output.detach()
                self.evaluations += 1
                return output
            cached = self.saved.get(index)
            if cached is None or (cached.shape, cached.dtype, cached.device) != (
                x.shape,
                x.dtype,
                x.device,
            ):
                raise RuntimeError("Invalid EVO residual cache")
            self.reuses += 1
            return cached

        return forward

    def reset(self, context):
        self.saved = {}
        self.step = -1
        self.query_open = self.in_forward = False
        self.evaluations = self.reuses = 0

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished EVO query")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.config.enabled:
            if (
                self.step + 1 != self.manifest["num_steps"]
                or self.evaluations != self.config.refresh_budget
            ):
                raise RuntimeError("Actual EVO execution differs from the frozen budget")
            self.backend.query_diagnostics.update(
                denoiser_calls=self.step + 1,
                evo_branch_evaluations=self.evaluations,
                evo_branch_reuses=self.reuses,
            )
        self.saved.clear()
        self.query_open = False

    def close(self):
        self.patches.close()
        if getattr(self.net, "_evb_evo", None) is self:
            del self.net._evb_evo
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="EVO global rollout-selected residual cache schedule",
            config=asdict(self.config),
            requires_training=False,
            reference_revision="09e0fcbdba14ebf7d8180f434dd5a015aeb0a71e",
            artifact_sha256=stable_hash(self.manifest),
            layout_sha256=stable_hash(self.layout),
            update_steps=self.layout,
            actual_branch_refreshes=self.config.refresh_budget,
            offline_frozen=self.manifest.get("complete", False),
            target_confirmed=self.selected.get("target_confirmed", False),
            population_sha256=self.manifest.get("population_sha256"),
            adaptation="24 branches; counted first/final refresh; one four-suite layout",
            excluded=[
                "absent cross-attention",
                "task-specific schedules",
                "Dev/final search feedback",
            ],
        )
