"""Artifact validation, runtime construction and offline calibration entrypoints."""


def preflight(config, resources):
    from .artifact import read_manifest

    if config.group_size != 128:
        raise ValueError("The pinned AWQ GEMM kernel requires group_size=128")
    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from .method import AWQ

    return AWQ(backend, config, resources.artifact_path)


def calibrate(backend, config, resources):
    from .calibrate import calibrate as run

    if config.group_size != 128:
        raise ValueError("The pinned AWQ GEMM kernel requires group_size=128")
    return run(backend, config, resources.dataset_path, resources.artifact_dir)
