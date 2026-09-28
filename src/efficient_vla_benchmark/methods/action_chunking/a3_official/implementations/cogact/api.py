def validate_experiment(experiment, config):
    if experiment.execution_horizon != 8:
        raise ValueError("A3 official-code compares adaptive prefixes against native H8")


def build(backend, config, resources):
    from .method import A3Official

    return A3Official(backend, config)
