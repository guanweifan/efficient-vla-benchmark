def build(backend, config, resources):
    from efficient_vla_backend_cogact.sampling import EpsilonSamplerContract

    from efficient_vla_benchmark.model_ops.scheduled_diffusion import ScheduledDiffusion

    from ...scheduler import make_scheduler

    return ScheduledDiffusion(
        backend,
        EpsilonSamplerContract.from_backend(backend),
        config,
        lambda betas: make_scheduler(betas, config.order),
        name="deis",
        reference="https://arxiv.org/abs/2204.13902",
        scope="DEIS log-rho multistep exponential integration with stable clean-endpoint limit",
    )
