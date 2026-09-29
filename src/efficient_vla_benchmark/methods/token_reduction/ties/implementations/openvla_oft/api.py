def validate_experiment(experiment, config):
    if config.refresh_steps % experiment.execution_horizon:
        raise ValueError("TIES refresh controls must align with the fixed execution horizon")


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config, "openvla_oft")


def build(backend, config, resources):
    from ...artifact import native_identity, read_manifest
    from .method import TIES

    manifest = read_manifest(resources.artifact_path, config, "openvla_oft")
    if native_identity(backend.metadata()) != manifest["native_identity"]:
        raise ValueError("TIES artifact native checkpoint/preprocessing identity mismatch")
    return TIES(backend, config, manifest)


def calibrate(backend, config, resources):
    from ...calibration import collect
    from .method import TIES

    return collect(backend, config, resources, TIES)
