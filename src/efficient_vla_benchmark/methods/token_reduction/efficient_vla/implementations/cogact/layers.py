from ...layer_pruning import LayerPruning, read_artifact


def preflight(config, resources):
    read_artifact(resources.artifact_path, config, "cogact")


def build(backend, config, resources):
    return LayerPruning(backend, backend.model_access.decoder, config, resources, "cogact")


def calibrate(backend, config, resources):
    from ...layer_pruning import calibrate as run

    return run(backend, backend.model_access.decoder, config, resources, "cogact")
