"""Lazy native decoder binding."""

PREFIX = "vlm.llm_backbone.llm.model.layers."


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config, resources.runtime_path)


def build(backend, config, resources):
    from ...method import DyQ

    return DyQ(backend, config, resources.artifact_path, PREFIX, resources.runtime_path)


def calibrate(backend, config, resources):
    from ...calibrate import calibrate as run

    return run(
        backend,
        config,
        resources.dataset_path,
        resources.artifact_dir,
        PREFIX,
        resources.runtime_path,
    )
