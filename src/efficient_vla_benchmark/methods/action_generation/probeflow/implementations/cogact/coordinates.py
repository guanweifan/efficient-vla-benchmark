"""Finite-noise DDIM <-> linear-mixture coordinates, without weight changes."""

import math

import numpy as np
import torch


class DDIMCoordinates:
    def __init__(self, diffusion, sampler):
        from action_model.gaussian_diffusion import ModelMeanType, ModelVarType

        if diffusion.model_mean_type != ModelMeanType.EPSILON or diffusion.model_var_type not in (
            ModelVarType.FIXED_SMALL,
            ModelVarType.FIXED_LARGE,
        ):
            raise ValueError("Bridge requires the native fixed-variance epsilon predictor")
        alphas = np.sqrt(np.asarray(diffusion.alphas_cumprod, dtype=np.float64))
        sigmas = np.sqrt(1 - alphas**2)
        self.r_grid = sigmas / (alphas + sigmas)
        if not np.isfinite(self.r_grid).all() or not np.all(np.diff(self.r_grid) > 0):
            raise ValueError("Invalid native noise schedule")
        self.start_index = int(sampler.timestep_map[-1])
        self.r_start = float(self.r_grid[self.start_index])
        self.initial_scale = float(alphas[self.start_index] + sigmas[self.start_index])
        if not 0 < self.r_start < 1:
            raise ValueError("A finite-noise native starting point is required")

    def coefficients(self, time):
        if not 0 <= time < 1:
            raise ValueError("Vector fields are evaluated before, not at, the clean endpoint")
        r = self.r_start * (1 - time)
        if r < self.r_grid[0] - 1e-12:
            raise ValueError("Requested probe grid extrapolates below native training timesteps")
        denominator = math.sqrt((1 - r) ** 2 + r**2)
        alpha, sigma = (1 - r) / denominator, r / denominator
        timestep = float(np.interp(r, self.r_grid, np.arange(len(self.r_grid))))
        return alpha, sigma, timestep

    def validate(self, config):
        self.coefficients(config.probe_time)
        self.coefficients(1 - 1 / config.max_steps)

    def velocity(self, model, y, time, kwargs):
        alpha, sigma, timestep = self.coefficients(time)
        z = y * (alpha + sigma)
        t = torch.full((z.shape[0],), timestep, device=z.device, dtype=torch.float32)
        epsilon = model(z, t, **kwargs)
        if epsilon.shape != z.shape:
            raise ValueError("Unexpected native epsilon geometry")
        x0 = (z - sigma * epsilon) / alpha
        # r = r_start * (1-u): noise->data time reverses the DDIM r direction.
        return self.r_start * (x0 - epsilon)
