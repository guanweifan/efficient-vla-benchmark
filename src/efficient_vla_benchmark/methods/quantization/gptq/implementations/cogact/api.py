def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from ...method import GPTQ

    return GPTQ(backend, config, resources.artifact_path, backend.model_access.decoder.layers)


def calibrate(backend, config, resources):
    from ...calibrate import calibrate as run

    return run(backend, config, resources, backend.model_access.decoder.layers, backend.model.cpu)
