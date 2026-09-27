"""AAC entropy elbow and cumulative movement floor, adapted from the MIT reference."""

import math

import numpy as np


def step_entropy(normalized, binary_gripper, regularization):
    """Raw normalized pose samples, plus empirical discrete gripper states."""
    x = np.asarray(normalized, dtype=np.float64)
    g = np.asarray(binary_gripper)
    if (
        x.ndim != 3
        or x.shape[0] < 4
        or x.shape[1] < 2
        or x.shape[2] != 7
        or g.shape != x.shape[:2]
        or not np.isfinite(x).all()
        or not np.isin(g, [0, 1]).all()
        or not math.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("AAC requires finite candidate chunks and binary gripper states")
    total = np.zeros(x.shape[1], dtype=np.float64)
    for start in (0, 3):
        for t in range(x.shape[1]):
            covariance = np.cov(x[:, t, start : start + 3], rowvar=False)
            covariance += regularization * np.eye(3)
            factor = np.linalg.cholesky(covariance)
            logdet = 2 * np.log(np.diag(factor)).sum()
            total[t] += 0.5 * (3 * np.log(2 * np.pi * np.e) + logdet)
    probability = np.clip(g.mean(axis=0), 1e-9, 1 - 1e-9)
    total -= probability * np.log(probability) + (1 - probability) * np.log(1 - probability)
    return total


def _axisangle_to_quaternion(rotation):
    angle = np.linalg.norm(rotation)
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    quaternion = np.zeros(4)
    quaternion[:3] = rotation / angle * np.sin(angle / 2)
    quaternion[3] = np.cos(angle / 2)
    return quaternion


def _quaternion_product(left, right):
    # Reference order: new rotation * accumulated rotation; float32 multiplication.
    x0, y0, z0, w0 = right
    x1, y1, z1, w1 = left
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def _axisangle_norm(quaternion):
    quaternion = quaternion.copy()
    quaternion[3] = np.clip(quaternion[3], -1, 1)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(denominator, 0.0):
        return 0.0
    axisangle = (quaternion[:3] * 2.0 * math.acos(quaternion[3])) / denominator
    return float(np.linalg.norm(axisangle))


def movement_profile(commands, gripper_weight):
    """Author code's cumulative translation, ordered rotation, and gripper toggle."""
    actions = np.asarray(commands)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("AAC movement needs finite seven-dimensional native commands")
    if not np.isin(actions[:, 6], [0, 1]).all():
        raise ValueError("AAC movement needs binary gripper states")
    magnitudes = []
    rotation = np.array([0.0, 0.0, 0.0, 1.0])
    for index, command in enumerate(actions):
        rotation = _quaternion_product(_axisangle_to_quaternion(command[3:6]), rotation)
        if index:
            position = np.linalg.norm(np.sum(actions[: index + 1, :3], axis=0))
            toggle = int(np.any(np.diff(actions[: index + 1, 6].astype(int))))
            magnitudes.append(position + _axisangle_norm(rotation) + gripper_weight * toggle)
    return np.asarray(magnitudes)


def select_prefix(normalized, commands, config):
    """Return the entropy elbow constrained by the reference movement lower bound."""
    normalized, commands = np.asarray(normalized), np.asarray(commands)
    if normalized.shape != commands.shape or len(normalized) != config.num_samples:
        raise ValueError("AAC candidate representations/configuration disagree")
    entropy = step_entropy(normalized, commands[:, :, 6], config.covariance_regularization)
    # Preserve the released implementation's prefix means and first-maximum ties.
    means = np.array([float(np.mean(entropy[:k])) for k in range(1, len(entropy) + 1)])
    elbow = max(int(np.argmax(np.diff(means))) + 1, 2)
    magnitude = movement_profile(commands[0], config.gripper_movement_weight)
    crossings = np.flatnonzero(magnitude > config.movement_threshold)
    floor = int(crossings[0]) + 2 if len(crossings) else len(entropy)
    return max(elbow, floor), dict(entropy_horizon=elbow, movement_horizon=floor)
