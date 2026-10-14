"""Per-query DDIM step context used by DiT activation-scale dispatch."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def ddim_schedule(action_model, num_steps: int) -> dict:
    """Identify the actual native sampler, including its floating-point noise schedule."""
    if getattr(action_model, "ddim_diffusion", None) is None:
        action_model.create_ddim(ddim_step=num_steps)
    diffusion = action_model.ddim_diffusion
    raw = getattr(diffusion, "timestep_map", None)
    original_steps = getattr(diffusion, "original_num_steps", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != num_steps:
        raise ValueError("CogACT DDIM timestep map has an invalid length")
    if (
        type(original_steps) is not int
        or original_steps < num_steps
        or any(type(t) is not int or t < 0 or t >= original_steps for t in raw)
        or list(raw) != sorted(set(raw))
    ):
        raise ValueError("CogACT DDIM timestep map must be unique, ascending original indices")
    betas = np.asarray(getattr(diffusion, "betas", None))
    if (
        betas.shape != (num_steps,)
        or not np.isfinite(betas).all()
        or not ((betas > 0) & (betas <= 1)).all()
    ):
        raise ValueError("CogACT DDIM noise schedule has invalid betas")
    noise_schedule = getattr(action_model, "noise_schedule", None)
    if not isinstance(noise_schedule, str) or not noise_schedule:
        raise ValueError("CogACT DDIM noise schedule name is missing")
    return {
        "sampler": "cogact_native_ddim",
        "index_space": "original_unscaled_diffusion_timestep",
        "timestep_map": list(raw),
        "row_timesteps": list(reversed(raw)),
        "row_zero": "first_denoising_call_highest_noise",
        "noise_schedule": noise_schedule,
        "original_num_steps": original_steps,
        "diffusion_steps": getattr(action_model, "diffusion_steps", None),
        "spaced_betas_sha256": hashlib.sha256(np.asarray(betas, dtype="<f8").tobytes()).hexdigest(),
    }


class DDIMStepTracker:
    """Attach to ``action_model.net`` without modifying the upstream model."""

    def __init__(self, action_model, num_steps: int):
        self.action_model = action_model
        self.num_steps = int(num_steps)
        self.schedule = ddim_schedule(action_model, self.num_steps)
        timestep_map = self.schedule["timestep_map"]
        # Diffusion loops run the map in reverse: highest noise is table row 0.
        self._step_by_timestep = {
            int(timestep): self.num_steps - 1 - index for index, timestep in enumerate(timestep_map)
        }
        # CogACT passes ``net.forward_with_cfg`` as a plain bound callable to
        # SpacedDiffusion.  That bypasses ``nn.Module.__call__`` and therefore
        # does not execute a hook registered directly on ``net``.  The
        # timestep embedding is entered by both the CFG and non-CFG paths, so
        # it is the stable hook point for the native timestep argument.
        timestep_embedder = getattr(action_model.net, "t_embedder", None)
        if timestep_embedder is None or not hasattr(timestep_embedder, "register_forward_pre_hook"):
            raise ValueError("CogACT DiT has no hookable timestep embedder")
        self._handle = timestep_embedder.register_forward_pre_hook(self._before_timestep)
        self.reset()

    def _before_timestep(self, _module, args):
        self.current_step = None
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Omega-QVLA could not observe the DiT timestep")
        values = args[0].reshape(-1)
        if (
            values.numel() == 0
            or values.dtype not in (torch.int32, torch.int64)
            or not bool((values == values[0]).all())
        ):
            raise RuntimeError("Omega-QVLA requires one uniform integer DiT timestep per call")
        timestep = int(values[0].item())
        step = self._step_by_timestep.get(timestep)
        if step is None:
            raise RuntimeError(f"Unsupported CogACT DDIM timestep {timestep}")
        self.current_step = step

    def step_for_timestep(self, timestep: int) -> int:
        """Return the calibration row used for a native DiT timestep.

        This public helper keeps the sampler-order convention testable without
        requiring a full CogACT forward.  Native DDIM calls expose original
        diffusion timesteps after ``SpacedDiffusion`` remapping; the first
        (highest-noise) call must therefore select row zero.
        """

        timestep = int(timestep)
        step = self._step_by_timestep.get(timestep)
        if step is None:
            raise ValueError(f"Unsupported CogACT DDIM timestep {timestep}")
        return step

    def reset(self) -> None:
        self.current_step = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.reset()
