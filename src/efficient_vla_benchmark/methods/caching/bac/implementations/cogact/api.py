def validate_experiment(experiment, config):
    if config.num_updates > experiment.backend_parameters.get("num_ddim_steps", 10):
        raise ValueError("BAC update budget exceeds native DDIM steps")
    if config.protected_blocks > 24:
        raise ValueError("CogACT DiT-B exposes 24 attention/MLP residual branches")


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from .method import BAC

    return BAC(backend, config, resources.artifact_path)


def calibrate(backend, config, resources):
    from .calibration import collect

    return collect(backend, config, resources)
