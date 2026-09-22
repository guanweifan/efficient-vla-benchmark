"""Native sixteen-step diffusion predictions; paper-time-aligned voting."""

from ...codec import decode, encode


def validate_experiment(experiment, config):
    if experiment.execution_horizon != 1:
        raise ValueError("VOTE's adjacent-time committee requires horizon=1 and a matched baseline")
    if config.max_history >= 16:
        raise ValueError(
            "CogACT's sixteen-step predictions support at most fifteen historical steps"
        )


def build(backend, config, resources):
    from ...method import VoteEnsemble

    return VoteEnsemble(config, encode=encode, decode=decode)
