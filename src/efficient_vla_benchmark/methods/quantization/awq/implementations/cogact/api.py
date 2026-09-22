"""Checksummed calibration and packed inference for the native CogACT decoder."""


def preflight(config, resources):
    from ...artifact import read_manifest

    if config.group_size != 128:
        raise ValueError("The pinned AWQ GEMM kernel requires group_size=128")
    manifest = read_manifest(resources.artifact_path, config)
    if manifest.get("backend") != "cogact":
        raise ValueError("AWQ artifact belongs to a different backend")


def build(backend, config, resources):
    from ...method import AWQ
    from .target import decoder_target

    return AWQ(backend, config, resources.artifact_path, decoder_target(backend))


def calibrate(backend, config, resources):
    from ...calibrate import calibrate as run
    from .target import decoder_target

    if config.group_size != 128:
        raise ValueError("The pinned AWQ GEMM kernel requires group_size=128")
    return run(
        backend, config, resources.dataset_path, resources.artifact_dir, decoder_target(backend)
    )
