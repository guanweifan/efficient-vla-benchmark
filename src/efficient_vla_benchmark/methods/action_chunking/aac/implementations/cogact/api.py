def validate_experiment(experiment, config):
    if experiment.execution_horizon != 8:
        raise ValueError("AAC is compared with the native H8 reference")


def build(backend, config, resources):
    from .method import AAC

    return AAC(backend, config)
