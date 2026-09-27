"""Reuse branch outputs while retaining current residual paths and native DDIM."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...artifact import read_manifest
from ...schedule import make_schedule


class BAC(InferenceMethod):
    name = "bac"

    def __init__(self, backend, config, artifact_path):
        self.backend, self.config = backend, config
        self.net = backend.model_access.action_denoiser
        self.manifest, similarities, errors = read_manifest(artifact_path, config)
        branches = [module for block in self.net.blocks for module in (block.attn, block.mlp)]
        if (
            self.net.training
            or self.manifest["checkpoint_sha256"] != backend._model_manifest["sha256"]
            or self.manifest["sampler_parameters"] != asdict(backend.parameters)
            or self.manifest["precision"] != str(next(self.net.parameters()).dtype)
            or self.manifest["num_blocks"] != len(branches)
            or self.manifest["num_steps"] != backend.parameters.num_ddim_steps
        ):
            raise ValueError("BAC artifact differs from native checkpoint/sampler/geometry")
        self.schedules, self.protected = make_schedule(
            similarities, errors, config.num_updates, config.protected_blocks
        )
        self.updates = [frozenset(schedule) for schedule in self.schedules]
        self.patches = ForwardPatches()
        self.closed = False
        self.reset(None)
        if config.enabled:
            if hasattr(self.net, "_evb_bac"):
                raise ValueError("BAC already attached")
            self.net._evb_bac = self
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
                raise RuntimeError("BAC requires nonnested inference queries")
            self.step += 1
            if self.step >= self.manifest["num_steps"]:
                raise RuntimeError("BAC received more denoising calls than calibrated")
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
                raise RuntimeError("BAC branch called outside native denoising")
            if self.step in self.updates[index]:
                result = original(x)
                self.saved[index] = result.detach()
                self.evaluations += 1
                return result
            cached = self.saved.get(index)
            if (
                cached is None
                or cached.shape != x.shape
                or cached.dtype != x.dtype
                or cached.device != x.device
            ):
                raise RuntimeError("Uninitialized or incompatible BAC residual cache")
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
            raise RuntimeError("Closed or unfinished BAC query")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.config.enabled:
            if self.step + 1 != self.manifest["num_steps"]:
                raise RuntimeError("Incomplete BAC denoising query")
            self.backend.query_diagnostics.update(
                denoiser_calls=self.step + 1,
                bac_branch_evaluations=self.evaluations,
                bac_branch_reuses=self.reuses,
                bac_protected_blocks=len(self.protected),
            )
        self.saved.clear()
        self.query_open = False

    def close(self):
        self.patches.close()
        if getattr(self.net, "_evb_bac", None) is self:
            del self.net._evb_bac
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="BAC ACS plus Bubbling Union on native attention/MLP residual branches",
            implementation_basis="official ACS interval objective and paper Eq. 13-14",
            reference_revision="82029a6fb0573fd07f4b26088219b0eb5ccc5a67",
            config=asdict(self.config),
            requires_training=False,
            artifact_sha256=self.manifest["statistics_sha256"],
            calibration_manifest_sha256=stable_hash(self.manifest),
            update_steps=[list(schedule) for schedule in self.schedules],
            protected_branches=list(self.protected),
            cache_lifetime="one native sampling query",
            cached_object="pre-residual attention and MLP outputs, CFG rows retained",
            calibration_scope="global four-suite training observations; no evaluation rollouts",
            excluded=["cross-attention absent in CogACT", "per-task tuned schedules"],
            adaptation="retain native norms and residual addition; final-step refresh",
        )
