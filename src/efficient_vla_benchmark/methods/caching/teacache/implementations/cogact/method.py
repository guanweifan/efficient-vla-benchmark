"""Full-stack residual reuse controlled by polynomial-rescaled timestep changes."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...artifact import read_manifest
from ...schedule import refresh_schedule


class TeaCache(InferenceMethod):
    name = "teacache"

    def __init__(self, backend, config, artifact_path):
        self.backend, self.config = backend, config
        self.net = backend.model_access.action_denoiser
        self.manifest, differences, coefficients = read_manifest(artifact_path, config)
        if (
            self.net.training
            or self.manifest["checkpoint_sha256"] != backend._model_manifest["sha256"]
            or self.manifest["sampler_parameters"] != asdict(backend.parameters)
            or self.manifest["precision"] != str(next(self.net.parameters()).dtype)
            or self.manifest["num_blocks"] != len(self.net.blocks)
            or self.manifest["num_steps"] != backend.parameters.num_ddim_steps
            or not all(
                hasattr(self.net, name)
                for name in (
                    "x_embedder",
                    "t_embedder",
                    "z_embedder",
                    "positional_embedding",
                    "final_layer",
                )
            )
        ):
            raise ValueError("TeaCache artifact differs from native checkpoint/sampler/geometry")
        self.schedule = refresh_schedule(
            differences.tolist(), coefficients.tolist(), config.threshold, config.warmup_steps
        )
        self.patches = ForwardPatches()
        self.closed = False
        self.active = config.enabled and not all(self.schedule)
        self.reset(None)
        if self.active:
            if any(
                hasattr(self.net, name) for name in ("_evb_denoiser_cache", "_evb_feature_cache")
            ):
                raise ValueError("Only one cache may own this denoiser")
            self.net._evb_feature_cache = self
            try:
                self.patches.wrap(self.net, self._denoise)
            except Exception:
                self.close()
                raise

    def _denoise(self, original):
        def forward(x, t, z):
            if not self.query_open or self.in_step:
                raise RuntimeError("TeaCache requires a nonnested native query")
            self.in_step = True
            try:
                if torch.is_grad_enabled() or self.net.training:
                    raise RuntimeError("TeaCache is inference-only")
                if self.step >= len(self.schedule):
                    raise RuntimeError("Unexpected extra denoiser call")
                if x.shape[0] != self.manifest["cfg_rows"] or not torch.equal(
                    t, t.new_full(t.shape, self.manifest["timesteps"][self.step])
                ):
                    raise RuntimeError("Native noise grid or CFG batch changed")
                signature = (tuple(x.shape), x.dtype, x.device, tuple(z.shape))
                if self.signature is not None and self.signature != signature:
                    raise RuntimeError("Denoiser geometry changed within a query")
                self.signature = signature
                hidden = self.net.x_embedder(x)
                time = self.net.t_embedder(t)
                condition = self.net.z_embedder(z, self.net.training)
                hidden = torch.cat((time.unsqueeze(1) + condition, hidden), dim=1)
                hidden = hidden + self.net.positional_embedding
                if self.schedule[self.step]:
                    result = hidden
                    for block in self.net.blocks:
                        result = block(result)
                    self.saved = (result - hidden).detach()
                    self.evaluations += 1
                else:
                    if self.saved is None or self.saved.shape != hidden.shape:
                        raise RuntimeError("Missing compatible TeaCache residual")
                    result = hidden + self.saved
                self.step += 1
                return self.net.final_layer(result)[:, 1:, :]
            except Exception:
                self.reset(None)
                raise
            finally:
                self.in_step = False

        return forward

    def reset(self, context):
        self.saved = self.signature = None
        self.step = self.evaluations = 0
        self.query_open = self.in_step = False

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished TeaCache query")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.active and self.step != len(self.schedule):
            self.reset(None)
            raise RuntimeError("Incomplete TeaCache native schedule")
        if self.active:
            self.backend.query_diagnostics.update(
                denoiser_calls=self.step,
                denoiser_evaluations=self.evaluations,
                teacache_reused_steps=self.step - self.evaluations,
            )
        self.saved = None
        self.query_open = False

    def close(self):
        self.patches.close()
        if getattr(self.net, "_evb_feature_cache", None) is self:
            del self.net._evb_feature_cache
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="TeaCache timestep-embedding variant; polynomial-gated full-stack residual reuse",
            reference_revision="7c10efc4702c6b619f47805f7abe4a7a08085aa0",
            implementation_basis="paper Eq. 4-7; official HiDream timestep-only signal and warmup",
            config=asdict(self.config),
            requires_training=False,
            artifact_sha256=self.manifest["statistics_sha256"],
            calibration_manifest_sha256=stable_hash(self.manifest),
            fresh_steps=list(self.schedule),
            schedule_scope="exact precomputation for fixed time-only signal, not uniform caching",
            cached_object="separate CFG rows of transformer-stack output minus current input",
            excluded=[
                "adaLN-modulated signal",
                "video scores",
                "input-adaptive noisy-token signal",
            ],
        )
