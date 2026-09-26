from ...layer_pruning import LayerPruning, read_artifact


def preflight(config, resources):
    read_artifact(resources.artifact_path, config, "openvla_oft")


def build(backend, config, resources):
    return LayerPruning(
        backend, backend.model.language_model.model, config, resources, "openvla_oft"
    )


def calibrate(backend, config, resources):
    from ...layer_pruning import calibrate as run

    return run(backend, backend.model.language_model.model, config, resources, "openvla_oft")
