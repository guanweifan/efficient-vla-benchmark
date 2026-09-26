"""Native sixteen-step predictions with H8 time-aligned two-candidate voting."""

from ...codec import decode, encode


def validate_experiment(experiment, config):
    if experiment.execution_horizon != 8:
        raise ValueError("VOTE chunk adaptation requires execution_horizon=8")
    if config.history_chunks * experiment.execution_horizon >= 16:
        raise ValueError(
            "CogACT H8 supports one overlapping historical chunk, not extra independent votes"
        )


def build(backend, config, resources):
    from ...method import VoteEnsemble

    return VoteEnsemble(config, encode=encode, decode=decode)
