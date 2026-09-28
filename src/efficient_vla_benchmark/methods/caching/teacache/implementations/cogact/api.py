def validate_experiment(experiment, config):
    steps = experiment.backend_parameters.get("num_ddim_steps", 10)
    if config.warmup_steps > steps or config.polynomial_degree >= steps - 1:
        raise ValueError("TeaCache warmup or polynomial degree exceeds the native noise grid")


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from .method import TeaCache

    return TeaCache(backend, config, resources.artifact_path)


def calibrate(backend, config, resources):
    from .calibration import collect

    return collect(backend, config, resources)
