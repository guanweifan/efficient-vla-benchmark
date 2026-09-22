def validate_experiment(experiment, config):
    steps = experiment.backend_parameters.get("num_ddim_steps", 10)
    if config.enabled and config.cache_interval > steps:
        raise ValueError("cache_interval cannot exceed the native DDIM call count")


def build(backend, config, resources):
    from .method import EfficientVLADiffusionCache

    return EfficientVLADiffusionCache(backend, config)
