"""EfficientVLA section 3.4: periodic per-block attention and MLP feature reuse."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod


class EfficientVLADiffusionCache(InferenceMethod):
    name = "efficient_vla_diffusion_cache"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.denoiser = backend.model_access.action_denoiser
        if self.denoiser.training or not getattr(self.denoiser, "blocks", None):
            raise ValueError("Feature caching requires a native eval-mode DiT with blocks")
        for block in self.denoiser.blocks:
            if not all(hasattr(block, name) for name in ("norm1", "attn", "norm2", "mlp")):
                raise ValueError("Unsupported DiT feature/residual structure")
        self.originals = []
        self.closed = False
        self.active = config.enabled and config.cache_interval > 1
        self.reset(None)
        if self.active:
            if getattr(self.denoiser, "_evb_feature_cache", None) is not None:
                raise ValueError("Only one feature cache may wrap the native DiT")
            self.denoiser._evb_feature_cache = self
            try:
                # forward_with_cfg calls forward directly, bypassing module pre-hooks.
                self._wrap(self.denoiser, self._denoise_wrapper)
                for index, block in enumerate(self.denoiser.blocks):
                    for kind in ("attn", "mlp"):
                        self._wrap(
                            getattr(block, kind),
                            lambda original, key=(index, kind): self._feature_wrapper(
                                original, key
                            ),
                        )
            except Exception:
                self.close()
                raise

    def _wrap(self, module, factory):
        previous = module.__dict__.get("forward")
        had_instance_forward = "forward" in module.__dict__
        wrapper = factory(module.forward)
        self.originals.append((module, had_instance_forward, previous))
        module.forward = wrapper

    def _denoise_wrapper(self, original):
        def forward(*args, **kwargs):
            if not self.query_open or self.in_step:
                raise RuntimeError("DiT feature caching requires a fresh nonnested policy query")
            if torch.is_grad_enabled() or self.denoiser.training:
                self.reset(None)
                raise RuntimeError("DiT feature caching is inference-only")
            self.in_step = True
            self.refresh = self.steps % self.config.cache_interval == 0
            self.visited = set()
            try:
                result = original(*args, **kwargs)
                expected = 2 * len(self.denoiser.blocks)
                if len(self.visited) != expected:
                    raise RuntimeError("Native DiT did not visit every attention/MLP boundary")
                self.refresh_steps += int(self.refresh)
                self.steps += 1
                return result
            except Exception:
                self.reset(None)
                raise
            finally:
                self.in_step = False

        return forward

    def _feature_wrapper(self, original, key):
        def forward(x, *args, **kwargs):
            if not self.in_step or key in self.visited:
                raise RuntimeError("Unexpected repeated or out-of-step DiT feature call")
            self.visited.add(key)
            signature = (tuple(x.shape), x.dtype, x.device)
            if self.refresh:
                output = original(x, *args, **kwargs)
                if not isinstance(output, torch.Tensor) or output.shape != x.shape:
                    raise ValueError("Expected a shape-preserving native DiT feature")
                self.cache[key] = (signature, output.detach())
                self.computed[key[1]] += 1
                return output
            saved = self.cache.get(key)
            if saved is None or saved[0] != signature:
                raise RuntimeError("DiT feature-cache geometry changed inside an action query")
            self.reused[key[1]] += 1
            return saved[1]

        return forward

    def reset(self, context):
        self.cache = {}
        self.steps = self.refresh_steps = 0
        self.computed = {"attn": 0, "mlp": 0}
        self.reused = {"attn": 0, "mlp": 0}
        self.query_open = self.in_step = self.refresh = False
        self.visited = set()

    def before_query(self, prepared):
        if self.closed:
            raise RuntimeError("The feature-cache method is closed")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.active:
            self.backend.query_diagnostics.update(
                denoiser_calls=self.steps,
                feature_refresh_steps=self.refresh_steps,
                feature_reuse_steps=self.steps - self.refresh_steps,
                attention_evaluations=self.computed["attn"],
                mlp_evaluations=self.computed["mlp"],
                attention_reuses=self.reused["attn"],
                mlp_reuses=self.reused["mlp"],
                cache_interval=self.config.cache_interval,
            )
        self.query_open = False
        self.cache.clear()

    def close(self):
        for module, had_instance, previous in reversed(self.originals):
            if had_instance:
                module.forward = previous
            else:
                del module.forward
        self.originals.clear()
        if getattr(self.denoiser, "_evb_feature_cache", None) is self:
            del self.denoiser._evb_feature_cache
        self.reset(None)
        self.closed = True

    def metadata(self):
        return {
            "scope": "EfficientVLA section 3.4 independent attention/MLP feature-cache core",
            "excluded": ["token pruning", "layer pruning", "sampling-step reduction"],
            "implementation_basis": "paper equations 6-11; independent native CogACT binding",
            "schedule": "first denoiser call, then every N calls within each policy query",
            "residuals": "live native inputs; attention and MLP outputs cached separately",
            "cfg": "native paired conditional/unconditional rows, never averaged in cache",
            "cache_lifetime": "one action query, cleared before and after sampling",
            "requires_training": False,
            "config": asdict(self.config),
        }
