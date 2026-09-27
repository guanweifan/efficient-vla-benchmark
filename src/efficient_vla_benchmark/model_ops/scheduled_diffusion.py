"""Optional finite-noise scheduler execution; no paper or backend selection."""

from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


def inference_grid(alphas_cumprod, start_index, steps, spacing):
    """Integer training indices, preserving both native finite-noise endpoints."""
    if not 2 <= steps <= start_index + 1:
        raise ValueError("Invalid inference budget")
    if spacing == "time_uniform":
        grid = np.linspace(start_index, 0, steps).round().astype(np.int64)
    elif spacing == "log_snr":
        alpha = np.asarray(alphas_cumprod, dtype=np.float64)[: start_index + 1]
        lambdas = 0.5 * np.log(alpha / (1 - alpha))
        target = np.linspace(lambdas[-1], lambdas[0], steps)
        grid = np.interp(target, lambdas[::-1], np.arange(start_index, -1, -1))
        grid = grid.round().astype(np.int64)
    else:
        raise ValueError("Unknown solver grid")
    if grid[0] != start_index or grid[-1] != 0 or not np.all(np.diff(grid) < 0):
        raise ValueError("Grid must be strictly decreasing without repeated model times")
    return grid


class ScheduledDiffusion(InferenceMethod):
    """Execute a caller-supplied scheduler through a native sampler contract."""

    def __init__(self, backend, contract, config, factory, *, name, reference, scope):
        if version("diffusers") != "0.30.3":
            raise ValueError("This scheduler bridge is validated for diffusers==0.30.3")
        self.backend, self.contract, self.config = backend, contract, config
        self.name, self.reference, self.scope = name, reference, scope
        self.patches = ForwardPatches()
        self.closed = False
        self.query_open = False
        self.scheduler = None
        self.grid = inference_grid(
            contract.alphas_cumprod, contract.start_index, config.num_steps, config.spacing
        )
        if config.num_steps > contract.native_steps:
            raise ValueError("This bounded acceleration port cannot exceed the native call budget")
        if config.enabled:
            if "ddim_sample_loop" in contract.sampler.__dict__:
                raise ValueError("A sampler replacement is already attached")
            self.scheduler = factory(contract.betas)
            self.patches.wrap(contract.sampler, self._loop, "ddim_sample_loop")

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
            if self.closed or not self.query_open or torch.is_grad_enabled():
                raise RuntimeError("Requires an active, nonnested inference query")
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
                raise ValueError("Unsupported native sampler options")
            rows = 2 if model_kwargs.get("cfg_scale", 1.0) > 1.0 else 1
            if noise.shape[0] != rows or model_kwargs["z"].shape[0] != rows:
                raise ValueError("Requires batch-one native paired conditioning")
            if rows == 2 and not torch.equal(noise[0], noise[1]):
                raise ValueError("Native CFG noise must be paired")

            scheduler = self.scheduler
            scheduler.set_timesteps(self.config.num_steps, device=noise.device)
            # Explicit pinned scheduler-state interface. No change to update equations.
            # Both the first noisy state and last prediction index stay native.
            alpha = self.contract.alphas_cumprod[self.grid]
            sigmas = np.sqrt((1 - alpha) / alpha)
            scheduler.timesteps = torch.as_tensor(self.grid, device=noise.device)
            scheduler.sigmas = torch.tensor(np.append(sigmas, 0.0), dtype=torch.float32)
            value = noise
            for timestep in scheduler.timesteps:
                times = timestep.expand(rows)
                epsilon = model(value, times, **model_kwargs)
                if epsilon.shape != value.shape:
                    raise ValueError("Denoiser changed native action geometry")
                value = scheduler.step(epsilon, timestep, value, return_dict=False)[0]
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite solver result")

            # Native eta-zero DDIM still draws discarded noise. Preserve its RNG
            # advance so future initial samples do not depend on the solver budget.
            for _ in range(self.contract.native_steps):
                torch.randn_like(noise)
            self.backend.query_diagnostics.update(
                denoiser_calls=len(self.grid),
                native_reference_steps=self.contract.native_steps,
                solver_order=self.config.order,
                solver_timesteps=self.grid.tolist(),
                solver_spacing=self.config.spacing,
                solver_native_start_index=self.contract.start_index,
                solver_clean_endpoint=True,
            )
            return value

        return sample

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished solver query")
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False

    def reset(self, context):
        self.query_open = False
        if self.scheduler is not None:
            self.scheduler.model_outputs = [None] * self.config.order
            if hasattr(self.scheduler, "last_sample"):
                self.scheduler.last_sample = None

    def close(self):
        self.patches.close()
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope=self.scope,
            paper=self.reference,
            config=asdict(self.config),
            requires_training=False,
            solver_runtime="diffusers==0.30.3",
            adaptation="native finite-noise start, explicit integer grid and clean endpoint",
            native_start_index=self.contract.start_index,
            inference_timesteps=self.grid.tolist(),
            cfg="unchanged native conditional/unconditional model callback",
            rng="native eta-zero discarded draws retained",
            exclusions=["retraining", "action-head replacement", "execution-horizon changes"],
        )
