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


