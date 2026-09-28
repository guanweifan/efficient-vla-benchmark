"""Official A3 masked denoising mapped to native CogACT DDIM coordinates."""

import numpy as np
import torch


def make_sampler(backend, steps):
    """Use the native executor on a thinned grid with unchanged noise endpoints."""
    from action_model.respace import SpacedDiffusion
    from efficient_vla_backend_cogact.sampling import EpsilonSamplerContract

    contract = EpsilonSamplerContract.from_backend(backend)
    native = contract.sampler
    if steps > contract.native_steps:
        raise ValueError("A3 denoising budget exceeds the configured native schedule")
    if steps == contract.native_steps:
        return native
    indices = np.rint(np.linspace(0, contract.native_steps - 1, steps)).astype(int)
    return SpacedDiffusion(
        use_timesteps={native.timestep_map[index] for index in indices},
        betas=contract.betas,
        model_mean_type=native.model_mean_type,
        model_var_type=native.model_var_type,
        loss_type=native.loss_type,
    )


def native_commands(samples, statistics):
    normalized = np.clip(np.asarray(samples), -1, 1)
    normalized[..., 6] = np.where(normalized[..., 6] < 0.5, 0, 1)
    low, high = np.asarray(statistics["q01"]), np.asarray(statistics["q99"])
    mask = np.asarray(statistics.get("mask", np.ones_like(low, dtype=bool)))
    if low.shape != (7,) or high.shape != (7,) or mask.shape != (7,) or mask[6]:
        raise ValueError("A3 requires seven-dimensional native deltas and binary gripper")
    result = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
    result[..., 6] = 1 - 2 * result[..., 6]
    return result


def conditional_ddim(sampler, model, noise, model_kwargs, fixed, mask):
    """Clamp known positions at both ends of EVERY original DDIM transition.

    GR00T code uses (1-t)*noise + t*fixed. CogACT uses the corresponding
    q(x_t|fixed, original_noise) on its native variance-preserving noise table.
    The clean endpoint is fixed exactly; no constant clean clamp at noisy steps.
    """
    if fixed.shape != noise.shape or mask.shape != noise.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Invalid conditional DDIM geometry")
    mask = mask.unsqueeze(-1)
    image = noise
    for index in reversed(range(sampler.num_timesteps)):
        timestep = torch.full((noise.shape[0],), index, device=noise.device, dtype=torch.long)
        known = sampler.q_sample(fixed, timestep, noise=noise)
        image = torch.where(mask, known, image)
        image = sampler.ddim_sample(
            model, image, timestep, clip_denoised=False, model_kwargs=model_kwargs, eta=0.0
        )["sample"]
        known_next = fixed if index == 0 else sampler.q_sample(fixed, timestep - 1, noise=noise)
        image = torch.where(mask, known_next, image)
    return image
