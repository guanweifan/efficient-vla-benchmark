"""Broadcast native pre-CFG denoiser predictions inside a profiled window."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


class DPCache(InferenceMethod):
    name = "dp_cache"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.model = backend.model_access.action_denoiser
        if self.model.training or not getattr(self.model, "blocks", None):
            raise ValueError("DP-Cache requires an eval-mode native DiT")
        self.patches = ForwardPatches()
        self.active = config.enabled and config.cache_interval > 1
        self.closed = False
        self.reset(None)
        if self.active:
            if hasattr(self.model, "_evb_denoiser_cache"):
                raise ValueError("Only one denoiser cache may be attached")
            self.model._evb_denoiser_cache = self
            self.patches.wrap(self.model, self._wrapper)

    def _wrapper(self, original):
        def forward(x, t, z):
            if not self.query_open or self.in_step:
                raise RuntimeError("DP-Cache requires a nonnested policy query")
            if torch.is_grad_enabled() or self.model.training:
                self.reset(None)
                raise RuntimeError("DP-Cache is inference-only")
            self.in_step = True
            try:
                signature = (tuple(x.shape), x.dtype, x.device, tuple(z.shape))
                if self.signature is not None and signature != self.signature:
                    raise RuntimeError("Denoiser geometry changed within a query")
                self.signature = signature
                c = self.config
                inside = c.start_step <= self.steps <= c.end_step
                refresh = not inside or (self.steps - c.start_step) % c.cache_interval == 0
                if refresh:
                    result = original(x, t, z)
                    self.saved = result.detach() if inside else None
                    self.evaluations += 1
                else:
                    if self.saved is None:
                        raise RuntimeError("Cannot reuse an uninitialized denoiser output")
                    result = self.saved
                    self.reuses += 1
                self.steps += 1
                return result
            except Exception:
                self.reset(None)
                raise
            finally:
                self.in_step = False

        return forward

    def reset(self, context):
        self.saved = self.signature = None
        self.query_open = self.in_step = False
        self.steps = self.evaluations = self.reuses = 0

    def before_query(self, prepared):
        if self.closed:
            raise RuntimeError("DP-Cache is closed")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.active:
            self.backend.query_diagnostics.update(
                denoiser_calls=self.steps,
                denoiser_evaluations=self.evaluations,
                denoiser_output_reuses=self.reuses,
                cache_start_step=self.config.start_step,
                cache_end_step=self.config.end_step,
            )
        self.query_open = False
        self.saved = None

    def close(self):
        self.patches.close()
        if getattr(self.model, "_evb_denoiser_cache", None) is self:
            del self.model._evb_denoiser_cache
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="DP-Cache stable-window output-broadcast core",
            excluded=["V-AEFusion", "compilation", "sampling-step deletion"],
            implementation_basis="paper section 5.2.2; independent CogACT DDIM port",
            cached_object="native pre-CFG epsilon output; conditional/unconditional rows separate",
            schedule="inclusive zero-based sampling-call window; native DDIM transitions unchanged",
            cache_lifetime="one policy query",
            requires_training=False,
            config=asdict(self.config),
        )
