"""Equations 6--9: choose before adding this query to the rolling window."""

import numpy as np


def select_prefix(variance, history, config, default_horizon):
    variance = np.asarray(variance, dtype=np.float64)
    if (
        variance.ndim != 1
        or variance.size < config.max_horizon
        or not np.isfinite(variance).all()
        or (variance < 0).any()
    ):
        raise ValueError("Expected finite nonnegative per-position denoising variance")
    if not history:
        # The paper does not define an empty-window threshold. Use the native
        # prefix once, then follow its causal decide-before-update ordering.
        return min(config.max_horizon, max(config.min_horizon, default_horizon)), None
    window = np.concatenate(history)
    if not np.isfinite(window).all() or (window < 0).any():
        raise ValueError("Invalid historical variance")
    threshold = float(window.mean() + config.alpha * window.std(ddof=0))
    crossings = np.flatnonzero(variance > threshold)
    horizon = int(crossings[0]) if crossings.size else config.max_horizon
    return min(config.max_horizon, max(config.min_horizon, horizon)), threshold
