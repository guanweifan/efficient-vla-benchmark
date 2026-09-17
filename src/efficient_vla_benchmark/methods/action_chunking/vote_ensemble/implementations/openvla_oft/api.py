import numpy as np


def encode(actions):
    """LIBERO environment commands -> paper's XYZ/rotation/0-1 gripper vectors."""
    result = np.asarray(actions, dtype=np.float64).copy()
    result[..., -1] = (1 - result[..., -1]) / 2
    return result


def decode(action):
    result = np.asarray(action).copy()
    result[-1] = -np.sign(2 * result[-1] - 1)
    return result


def validate_experiment(experiment, config):
    if experiment.execution_horizon != 1:
        raise ValueError("VOTE ensemble requires execution_horizon=1 and a matched baseline")
    if config.max_history >= 8:
        raise ValueError("OFT's eight-step predictions support at most seven historical steps")


def build(backend, config, resources):
    from ...method import VoteEnsemble

    return VoteEnsemble(config, encode=encode, decode=decode)
