"""Native sampler geometry and lifecycle, independent of solver selection."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpsilonSamplerContract:
    sampler: object
    betas: np.ndarray
    alphas_cumprod: np.ndarray
    start_index: int
    native_steps: int

    @classmethod
    def from_backend(cls, backend):
        from action_model.gaussian_diffusion import ModelMeanType, ModelVarType

        action = backend.model.action_model
        if action.net.training:
            raise ValueError("Sampler replacement requires an eval-mode action head")
        if action.ddim_diffusion is None:
            action.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
        native, full = action.ddim_diffusion, action.diffusion
        if full.model_mean_type != ModelMeanType.EPSILON or full.model_var_type not in (
            ModelVarType.FIXED_SMALL,
            ModelVarType.FIXED_LARGE,
        ):
            raise ValueError("Requires fixed-variance epsilon prediction")
        if native.num_timesteps != backend.parameters.num_ddim_steps:
            raise ValueError("Native schedule differs from configured reference")
        return cls(
            native,
            np.asarray(full.betas, dtype=np.float64).copy(),
            np.asarray(full.alphas_cumprod, dtype=np.float64).copy(),
            int(native.timestep_map[-1]),
            native.num_timesteps,
        )
