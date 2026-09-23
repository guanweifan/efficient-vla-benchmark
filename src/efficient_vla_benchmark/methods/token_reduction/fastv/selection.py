"""Independent implementation of the released last-query top-k selection."""

import torch


def attention_selection(probabilities, visual, keep_ratio):
    if probabilities.ndim != 1 or probabilities.shape != visual.shape:
        raise ValueError("Expected one head-averaged query row aligned with current tokens")
    ids = visual.nonzero().flatten()
    if not ids.numel():
        raise ValueError("No visual keys")
    count = max(1, round(ids.numel() * keep_ratio))
    selected = torch.argsort(probabilities[ids], descending=True, stable=True)[:count]
    keep = ~visual
    keep[ids[selected]] = True
    return keep
