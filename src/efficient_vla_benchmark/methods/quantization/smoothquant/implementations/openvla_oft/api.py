def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from ...method import SmoothQuant

    return SmoothQuant(
        backend, config, resources.artifact_path, backend.model.language_model.model.layers
    )


def calibrate(backend, config, resources):
    from ...calibrate import calibrate as run

    return run(backend, config, resources, backend.model.language_model.model.layers)
