"""Prominent low-speed valleys in native Cartesian delta commands."""

import numpy as np
from scipy.signal import find_peaks


def instruction_key(text):
    return " ".join(text.lower().replace("_", " ").split())


def speed_profile(actions, config):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("PACE binding requires finite native seven-dimensional commands")
    # These commands are already displacement increments, NOT positions.
    # Differencing them would estimate acceleration instead of the paper's speed.
    speed = np.linalg.norm(actions[:, :3], axis=-1)
    half = config.smoothing_window // 2
    if len(speed) < config.smoothing_window:
        raise ValueError("Predicted profile is shorter than the smoothing window")
    padded = np.pad(speed, (half, half), mode="edge")
    return np.convolve(
        padded, np.ones(config.smoothing_window) / config.smoothing_window, mode="valid"
    )


def valleys(actions, config):
    profile = speed_profile(actions, config)
    positions, properties = find_peaks(
        -profile, distance=config.minimum_spacing, prominence=(None, None)
    )
    # Position k is the low-speed kth command; execute through it before replanning.
    return positions + 1, properties["prominences"]


def select_prefix(actions, threshold, config):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("PACE threshold must be finite and nonnegative")
    boundaries, prominences = valleys(actions, config)
    maximum = len(actions) if config.max_horizon is None else config.max_horizon
    if maximum > len(actions):
        raise ValueError("PACE cannot extend native predictions")
    accepted = boundaries[(prominences >= threshold) & (boundaries <= maximum)]
    return int(accepted.min()) if len(accepted) else maximum, len(accepted)
