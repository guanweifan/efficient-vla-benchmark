def validate_experiment(experiment, config):
    steps = experiment.backend_parameters.get("num_ddim_steps", 10)
    if config.end_step >= steps or config.cache_interval > steps:
        raise ValueError("DP-Cache window/interval exceeds native denoising schedule")


def build(backend, config, resources):
    from .method import DPCache

    return DPCache(backend, config)
