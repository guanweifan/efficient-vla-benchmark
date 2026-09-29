def validate_experiment(experiment, config):
    if experiment.execution_horizon != config.reference_horizon or config.reference_horizon != 8:
        raise ValueError("This Falcon binding requires native H8 execution")


def build(backend, config, resources):
    from efficient_vla_backend_cogact.sampling import EpsilonSamplerContract

    from ...method import Falcon

    return Falcon(backend, EpsilonSamplerContract.from_backend(backend), config)
