"""ProbeFlow's complete adaptive rule through an explicitly disclosed DDIM bridge."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...integration import integrate
from .coordinates import DDIMCoordinates


class ProbeFlow(InferenceMethod):
    name = "probeflow"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.patches = ForwardPatches()
        self.closed = False
        self.query_open = False
        self.sampler = None
        if config.enabled:
            action = backend.model.action_model
            if action.net.training:
                raise ValueError("ProbeFlow requires an eval-mode action head")
            if action.ddim_diffusion is None:
                action.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
            self.sampler = action.ddim_diffusion
            self.coordinates = DDIMCoordinates(action.diffusion, self.sampler)
            self.coordinates.validate(config)
            if hasattr(self.sampler, "_evb_probeflow"):
                raise ValueError("ProbeFlow already attached")
            self.sampler._evb_probeflow = self
            self.patches.wrap(self.sampler, self._loop, "ddim_sample_loop")

    def _loop(self, original):
        def sample(
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
        ):
            if not self.query_open or torch.is_grad_enabled():
                raise RuntimeError("ProbeFlow requires a nonnested inference query")
            if (
                noise is None
                or tuple(shape) != tuple(noise.shape)
                or noise.ndim != 3
                or clip_denoised
                or denoised_fn is not None
                or cond_fn is not None
                or progress
                or eta != 0
                or not isinstance(model_kwargs, dict)
                or "z" not in model_kwargs
            ):
                raise ValueError("Unsupported nonnative DDIM options")
            rows = 2 if model_kwargs.get("cfg_scale", 1.0) > 1.0 else 1
            if noise.shape[0] != rows or model_kwargs["z"].shape[0] != rows:
                raise ValueError("Only native batch-one paired CFG is supported")
            if rows == 2 and not torch.equal(noise[0], noise[1]):
                raise ValueError("CFG noises must be paired")

            def velocity(y, time):
                # The native loop callback already refers to the original DiT
                # (not the SpacedDiffusion timestep wrapper).
                return self.coordinates.velocity(model, y, time, model_kwargs)

            initial = noise / self.coordinates.initial_scale
            result, diagnostics = integrate(velocity, initial, self.config)
            # Preserve the matched native eta-zero sampler's discarded random
            # draws, keeping later query initial noises common across budgets.
            for _ in range(self.sampler.num_timesteps):
                torch.randn_like(noise)
            self.backend.query_diagnostics.update(diagnostics)
            self.backend.query_diagnostics["native_reference_steps"] = self.sampler.num_timesteps
            return result  # At r=0, linear-mixture coordinates equal normalized actions.

        return sample

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished ProbeFlow query")
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False

    def reset(self, context):
        self.query_open = False

    def close(self):
        self.patches.close()
        if self.sampler is not None and getattr(self.sampler, "_evb_probeflow", None) is self:
            del self.sampler._evb_probeflow
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope=(
                "diagnostic fixed Euler, NOT ProbeFlow"
                if self.config.diagnostic_fixed_steps
                else "ProbeFlow lookahead cosine probe, adaptive budget and evaluation reuse"
            ),
            paper="https://arxiv.org/html/2603.17850v1",
            implementation_basis="equations 5-11 and Algorithm 1; no official runtime claimed",
            config=asdict(self.config),
            requires_training=False,
            adaptation="native finite-noise epsilon model mapped to linear-mixture ODE coordinates",
            interpolation="fractional timestep; linear interpolation in r=sigma/(alpha+sigma)",
            excluded=["claim of FM-trained CogACT", "learned shortcut or distillation", "OFT L1"],
        )
