def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config, "openvla_oft")


def build(backend, config, resources):
    from ...method import EcoVLA

    return EcoVLA(
        backend,
        backend.model.language_model.model,
        backend.model.vision_backbone,
        config,
        resources,
        "openvla_oft",
        causal=False,
    )


def calibrate(backend, config, resources):
    from ...artifact import calibrate as run

    return run(backend, backend.model.language_model.model, config, resources, "openvla_oft")
