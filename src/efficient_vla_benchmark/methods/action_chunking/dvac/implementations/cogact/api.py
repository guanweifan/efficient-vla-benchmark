def validate_experiment(experiment, config):
    steps = experiment.backend_parameters.get("num_ddim_steps", 10)
    if config.tail_steps > steps or config.max_horizon > 16:
        raise ValueError("DVAC bounds exceed native CogACT denoising/prediction lengths")


def build(backend, config, resources):
    from .method import DVAC

    return DVAC(backend, config)
