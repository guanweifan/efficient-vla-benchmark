def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)


def validate_experiment(experiment, config):
    if config.refinement_steps >= experiment.backend_parameters.get("num_ddim_steps", 10):
        raise ValueError("ActionCache refinement must use fewer than all native steps")


def build(backend, config, resources):
    from .method import ActionCache

    return ActionCache(backend, config, resources.artifact_path)
