def preflight(config, resources):
    from ...artifact import load_artifact

    load_artifact(resources.artifact_path, config)


def build(backend, config, resources):
    from ...artifact import load_artifact
    from .method import EVO

    return EVO(backend, config, load_artifact(resources.artifact_path, config))
