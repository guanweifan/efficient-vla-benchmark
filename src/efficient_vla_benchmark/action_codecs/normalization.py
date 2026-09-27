"""Calibration-only quantile bounds and reversible raw-gripper conventions."""

import numpy as np


def fit_quantiles(corpus):
    result = {}
    suites = sorted({r["suite"] for r in corpus.rows("calibration")})
    for suite in suites:
        arrays = [corpus.window(r)[0] for r in corpus.rows("calibration") if r["suite"] == suite]
        actions = np.concatenate(arrays)
        low, high = np.quantile(actions[:, :6], [0.01, 0.99], axis=0)
        result[suite] = {"low": low.tolist(), "high": high.tolist()}
    return result


def normalize(actions, bounds, *, signed_gripper):
    actions = np.array(actions, dtype=np.float32, copy=True)
    low, high = (np.array(bounds[k], dtype=np.float32) for k in ("low", "high"))
    actions[:, :6] = np.clip(2 * (actions[:, :6] - low) / (high - low + 1e-8) - 1, -1, 1)
    actions[:, :6][:, high == low] = 0
    opened = 1 - np.clip(actions[:, 6], 0, 1)
    actions[:, 6] = 2 * opened - 1 if signed_gripper else opened
    return actions


def unnormalize(actions, bounds, *, signed_gripper):
    actions = np.array(actions, dtype=np.float32, copy=True)
    low, high = (np.array(bounds[k], dtype=np.float32) for k in ("low", "high"))
    actions[:, :6] = (actions[:, :6] + 1) * 0.5 * (high - low + 1e-8) + low
    # Keep continuous gripper reconstruction; do not improve MSE by thresholding it.
    opened = (actions[:, 6] + 1) * 0.5 if signed_gripper else actions[:, 6]
    actions[:, 6] = 1 - 2 * opened
    return actions
