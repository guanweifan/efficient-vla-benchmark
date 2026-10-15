def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def build(backend, config, resources):
    from ...method import GPTQ

    return GPTQ(backend, config, resources.artifact_path, backend.model.language_model.model.layers)


def calibrate(backend, config, resources):
    from ...calibrate import calibrate as run

    def offload():
        backend.model.cpu()
        backend.action_head.cpu()
        backend.proprio_projector.cpu()

    return run(backend, config, resources, backend.model.language_model.model.layers, offload)
