"""LIBERO observation and action conventions; no simulator imports."""

import numpy as np
from PIL import Image


def prepare_observation(observation):
    observation.validate()
    # The common environment adapter has already oriented the RGB image.
    # Native CogACT owns resizing/normalization. This model uses only agentview.
    return {
        "image": Image.fromarray(observation.image),
        "instruction": observation.instruction,
        "observation": observation,
    }


def process_actions(actions):
    result = np.asarray(actions, dtype=np.float64).copy()
    if result.shape != (16, 7) or not np.isfinite(result).all():
        raise ValueError("CogACT must return sixteen finite seven-dimensional actions")
    if not np.isin(result[:, -1], [0.0, 1.0]).all():
        raise ValueError("Expected native binary open-gripper actions")
    # Native/OXE: 1=open, 0=closed. LIBERO: -1=open, +1=closed.
    result[:, -1] = 1.0 - 2.0 * result[:, -1]
    return result
