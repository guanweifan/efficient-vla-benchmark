def validate_experiment(experiment, config):
    steps = experiment.backend_parameters.get("num_ddim_steps", 10)
    if config.denoising_step > steps or config.max_horizon > 16 or config.run_length > 16:
        raise ValueError("AutoHorizon exceeds native CogACT denoising/prediction lengths")


def build(backend, config, resources):
    from .method import AutoHorizon

    return AutoHorizon(backend, config)
