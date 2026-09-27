"""Invertible command-space embedding, not an oracle trajectory of future poses."""

import numpy as np
from scipy.spatial.transform import Rotation

from ...core import actions_checked


def observed_reference(context):
    state = context.initial_state
    value = np.zeros(26, dtype=np.float32)
    value[:3] = state[:3]
    value[3:9] = Rotation.from_rotvec(state[3:6]).as_matrix()[:2].reshape(6)
    # Pinned Panda XML: first finger 0..0.04, second -0.04..0 metres.
    value[9] = np.clip((state[6] - state[7]) / 0.08, 0, 1)
    return value


def command_path(actions, context):
    actions = actions_checked(actions)
    if np.any(np.abs(actions[:, :6]) > 1.000001):
        raise ValueError("Command-space embedding expects the native bounded OSC controls")
    reference = observed_reference(context)
    result = np.zeros((len(actions), 26), dtype=np.float32)
    result[:, :3] = reference[:3] + np.cumsum(actions[:, :3].astype(np.float64) * 0.05, axis=0)
    rotation = Rotation.from_rotvec(context.initial_state[3:6]).as_matrix()
    # Native OSC uses axis-angle increments left-multiplied onto current orientation.
    for i, delta in enumerate(actions[:, 3:6]):
        rotation = Rotation.from_rotvec(delta.astype(np.float64) * 0.5).as_matrix() @ rotation
        result[i, 3:9] = rotation[:2].reshape(6)
    result[:, 9] = (1 - actions[:, 6]) * 0.5
    return result, reference


def raw_commands(path, context):
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 26 or not np.isfinite(path[:, :10]).all():
        raise ValueError("Invalid canonical action path")
    action = np.zeros((len(path), 7), dtype=np.float32)
    positions = np.concatenate([context.initial_state[None, :3], path[:, :3]])
    action[:, :3] = np.diff(positions, axis=0) / 0.05
    previous = Rotation.from_rotvec(context.initial_state[3:6]).as_matrix()
    for i, r6 in enumerate(path[:, 3:9]):
        x = r6[:3]
        nx = np.linalg.norm(x)
        if nx < 1e-8:
            raise ValueError("Degenerate decoded orientation; no identity fallback")
        x = x / nx
        y = r6[3:] - np.dot(r6[3:], x) * x
        ny = np.linalg.norm(y)
        if ny < 1e-8:
            raise ValueError("Degenerate decoded rotation basis")
        y = y / ny
        current = np.stack([x, y, np.cross(x, y)])
        action[i, 3:6] = Rotation.from_matrix(current @ previous.T).as_rotvec() / 0.5
        previous = current
    action[:, 6] = 1 - 2 * path[:, 9]
    return action
