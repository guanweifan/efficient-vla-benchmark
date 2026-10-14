"""Lazy binding; common harness remains algorithm-independent."""


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from .method import HoloQ

    return HoloQ(backend, config, resources.artifact_path)


def calibrate(backend, config, resources):
    from .calibrate import calibrate as run

    return run(backend, config, resources.dataset_path, resources.artifact_dir)
