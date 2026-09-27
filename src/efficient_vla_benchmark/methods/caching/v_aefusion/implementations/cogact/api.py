def validate_experiment(experiment, config):
    steps = experiment.backend_parameters.get("num_ddim_steps", 10)
    if config.stale_steps >= steps:
        raise ValueError("V-AEFusion requires at least one fresh denoising step")


def build(backend, config, resources):
    from .method import VAEFusion

    return VAEFusion(backend, config)
