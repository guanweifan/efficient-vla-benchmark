VISION_PREFIX = "vision_backbone."
LANGUAGE_PREFIX = "language_model.model.layers."


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from ...method import EaqVLA

    return EaqVLA(backend, config, resources.artifact_path, VISION_PREFIX, LANGUAGE_PREFIX)


def calibrate(backend, config, resources):
    from ...calibrate import calibrate as run

    return run(
        backend,
        config,
        resources.dataset_path,
        resources.artifact_dir,
        VISION_PREFIX,
        LANGUAGE_PREFIX,
    )
