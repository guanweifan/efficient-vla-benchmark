"""Native eta-zero DDIM transition with an explicit private RNG.

Equations and arithmetic order follow the pinned CogACT GaussianDiffusion
implementation (itself derived from OpenAI improved-diffusion, MIT).
Only the otherwise unused eta-zero noise draw uses the supplied generator.
"""

import torch


def private_ddim_step(sampler, model, x, t, kwargs, generator):
    from action_model.gaussian_diffusion import _extract_into_tensor

    out = sampler.p_mean_variance(model, x, t, clip_denoised=False, model_kwargs=kwargs)
    eps = sampler._predict_eps_from_xstart(x, t, out["pred_xstart"])
    alpha = _extract_into_tensor(sampler.alphas_cumprod, t, x.shape)
    previous = _extract_into_tensor(sampler.alphas_cumprod_prev, t, x.shape)
    sigma = 0.0 * torch.sqrt((1 - previous) / (1 - alpha)) * torch.sqrt(1 - alpha / previous)
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    mean = out["pred_xstart"] * torch.sqrt(previous) + torch.sqrt(1 - previous - sigma**2) * eps
    mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
    return mean + mask * sigma * noise
