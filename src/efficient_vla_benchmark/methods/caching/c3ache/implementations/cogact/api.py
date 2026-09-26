def validate_experiment(experiment, config):
    if config.cache_steps >= experiment.backend_parameters.get("num_ddim_steps", 10):
        raise ValueError("C3ache must retain at least one fully computed tail step")


def build(backend, config, resources):
    from .method import C3ache

    return C3ache(backend, config)
