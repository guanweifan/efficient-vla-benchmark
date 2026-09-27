def build(backend, config, resources):
    from efficient_vla_backend_cogact.sampling import EpsilonSamplerContract

    from efficient_vla_benchmark.model_ops.scheduled_diffusion import ScheduledDiffusion

    from ...scheduler import make_scheduler

    return ScheduledDiffusion(
        backend,
        EpsilonSamplerContract.from_backend(backend),
        config,
        lambda betas: make_scheduler(betas, config.order),
        name="unipc",
        reference="https://arxiv.org/abs/2302.04867",
        scope="UniPC bh2 predictor-corrector integration with recycled evaluations",
    )
