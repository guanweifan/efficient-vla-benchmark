def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def validate_experiment(experiment, config):
    if config.smoothing_window > 8 or (config.max_horizon is not None and config.max_horizon > 8):
        raise ValueError("PACE exceeds this backend's native prediction length")


def build(backend, config, resources):
    from ...method import PACE

    return PACE(config, resources.artifact_path, prediction_length=8)
