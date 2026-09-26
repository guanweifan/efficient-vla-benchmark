"""Same-step, cross-chunk suffix residuals with live conditioning and tail refresh."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


class C3ache(InferenceMethod):
    name = "c3ache_conditioned_residual"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.model = backend.model_access.action_denoiser
        self.blocks = self.model.blocks
        if self.model.training or not 1 <= config.live_prefix_blocks < len(self.blocks):
            raise ValueError(
                "C3ache requires a nonempty live prefix and cached suffix in eval mode"
            )
        self.patches = ForwardPatches()
        self.closed = False
        self.active = config.enabled and config.refresh_interval > 1
        self.reset(None)
        if self.active:
            if hasattr(self.model, "_evb_denoiser_cache"):
                raise ValueError("Only one denoiser cache may be attached")
            self.model._evb_denoiser_cache = self
            try:
                self.patches.wrap(self.model, self._denoiser)
                for i, block in enumerate(self.blocks):
                    if i >= config.live_prefix_blocks:
                        self.patches.wrap(block, lambda original, i=i: self._block(original, i))
            except Exception:
                self.close()
                raise

    def _denoiser(self, original):
        def forward(x, t, z):
            if not self.query_open or self.in_step:
                raise RuntimeError("C3ache requires a nonnested policy query")
            if torch.is_grad_enabled() or self.model.training:
                self.reset(None)
                raise RuntimeError("C3ache is inference-only")
            self.in_step = True
            self.visited = 0
            self.use_cache = (
                self.step < self.config.cache_steps
                and self.chunk % self.config.refresh_interval != 0
            )
            try:
                result = original(x, t, z)
                if self.visited != len(self.blocks) - self.config.live_prefix_blocks:
                    raise RuntimeError("Incomplete DiT suffix traversal")
                self.reuses += int(self.use_cache)
                self.step += 1
                return result
            except Exception:
                self.reset(None)
                raise
            finally:
                self.in_step = False
                self.anchor = None

        return forward

    def _block(self, original, index):
        def forward(x):
            if not self.in_step:
                raise RuntimeError("C3ache block outside denoiser call")
            self.visited += 1
            signature = (tuple(x.shape), x.dtype, x.device)
            if index == self.config.live_prefix_blocks:
                self.anchor = x
            if self.use_cache:
                if self.step not in self.cache or self.cache[self.step][0] != signature:
                    raise RuntimeError("C3ache cache geometry or sampling schedule changed")
                if index == len(self.blocks) - 1:
                    return x + self.cache[self.step][1]
                return x
            output = original(x)
            if index == len(self.blocks) - 1 and self.step < self.config.cache_steps:
                self.cache[self.step] = (signature, (output - self.anchor).detach())
            return output

        return forward

    def reset(self, context):
        self.cache = {}
        self.chunk = self.step = self.reuses = 0
        self.query_open = self.in_step = self.use_cache = False
        self.anchor = None
        self.expected_steps = None

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished C3ache query")
        self.step = self.reuses = 0
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.active:
            if self.step <= self.config.cache_steps:
                self.reset(None)
                raise RuntimeError("C3ache tail step was not evaluated")
            if self.expected_steps not in (None, self.step):
                self.reset(None)
                raise RuntimeError("Sampling schedule changed across chunks")
            self.expected_steps = self.step
            self.backend.query_diagnostics.update(
                denoiser_calls=self.step,
                residual_reuse_steps=self.reuses,
                live_prefix_blocks=self.config.live_prefix_blocks,
                cached_suffix_blocks=len(self.blocks) - self.config.live_prefix_blocks,
                cache_entries=len(self.cache),
                cache_chunk_index=self.chunk,
            )
        self.chunk += 1
        self.query_open = False

    def close(self):
        self.patches.close()
        if getattr(self.model, "_evb_denoiser_cache", None) is self:
            del self.model._evb_denoiser_cache
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="C3ache cross-chunk same-step residual core with live conditioning prefix",
            implementation_basis="paper equations 6-8; independent CogACT boundary adaptation",
            excluded=["Fast-WAM architecture", "all-block skipping", "within-chunk caching"],
            boundary="h_L - h_prefix; fresh prefix mixes latest cognition into action rows",
            schedule="first chunk and periodic full refresh; complete tail denoising steps",
            cfg="native conditional/unconditional rows retained independently",
            requires_training=False,
            config=asdict(self.config),
        )
