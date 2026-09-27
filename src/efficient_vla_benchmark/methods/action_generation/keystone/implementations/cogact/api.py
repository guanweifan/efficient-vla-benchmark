def validate_experiment(experiment, config):
    if config.selection_horizon != experiment.execution_horizon:
        raise ValueError("Select over the actual executed action prefix")
    if config.selection_horizon > 16:
        raise ValueError("Selection prefix exceeds native CogACT predictions")


def build(backend, config, resources):
    from .method import KeyStone

    return KeyStone(backend, config)
