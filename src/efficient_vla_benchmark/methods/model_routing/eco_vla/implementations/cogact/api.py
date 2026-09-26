def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config, "cogact")


def build(backend, config, resources):
    from ...method import EcoVLA

    return EcoVLA(
        backend,
        backend.model_access.decoder,
        backend.model_access.vision,
        config,
        resources,
        "cogact",
        causal=True,
    )


def calibrate(backend, config, resources):
    from ...artifact import calibrate as run

    return run(backend, backend.model_access.decoder, config, resources, "cogact")
