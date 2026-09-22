from ...codec import decode, encode


def validate_experiment(experiment, config):
    if experiment.execution_horizon != 1:
        raise ValueError("VOTE ensemble requires execution_horizon=1 and a matched baseline")
    if config.max_history >= 8:
        raise ValueError("OFT's eight-step predictions support at most seven historical steps")


def build(backend, config, resources):
    from ...method import VoteEnsemble

    return VoteEnsemble(config, encode=encode, decode=decode)
