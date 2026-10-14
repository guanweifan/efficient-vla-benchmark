"""Lazy construction of calibrated and floating-control author runtimes."""


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config, resources.runtime_path)


def build(backend, config, resources):
    from .method import ActQuant

    return ActQuant(backend, config, resources)


def calibrate(backend, config, resources):
    from .calibrate import calibrate as run

    return run(
        backend, config, resources.dataset_path, resources.artifact_dir, resources.runtime_path
    )
