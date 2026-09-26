"""Equations 5-11, with distinct fusion and history-update coefficients."""

import math
from collections import deque

import numpy as np
import torch
from torch.nn import functional as F


class SparsityTrigger:
    def __init__(self, quantile, window):
        self.quantile, self.window = quantile, window
        self.reset()

    def reset(self):
        self.previous = None
        self.history = deque(maxlen=self.window)

    def update(self, features):
        features = features.detach()
        if self.previous is None or self.previous.shape != features.shape:
            self.reset()
            dense = True
            similarity = None
        else:
            similarity = float(
                F.cosine_similarity(features.float(), self.previous.float(), dim=-1).mean()
            )
            # Equation 6 excludes the current similarity from the threshold.
            dense = not self.history or similarity < float(np.quantile(self.history, self.quantile))
            self.history.append(similarity)
        self.previous = features.clone()
        return dense, similarity


def importance(weight_norm, history, current, alpha, momentum):
    fused = alpha * history + (1 - alpha) * current
    updated = momentum * history + (1 - momentum) * current
    # || |W[:, k]|^2 E[k] ||_2 = E[k] sqrt(sum_i W[i,k]^4).
    return weight_norm * fused, updated


def select_channels(scores, keep_ratio, *, group_size=1, alignment=1):
    groups = scores.shape[-1] // group_size
    if groups * group_size != scores.shape[-1]:
        raise ValueError("Structured channels must form complete groups")
    if group_size > 1:
        # Coupled attention heads are indivisible; use the group's joint L2 score.
        scores = scores.reshape(*scores.shape[:-1], groups, group_size).norm(dim=-1)
    count = min(groups, max(1, math.ceil(groups * keep_ratio / alignment) * alignment))
    chosen = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
    chosen = chosen.sort(dim=-1).values
    if group_size == 1:
        return chosen
    offsets = torch.arange(group_size, device=chosen.device)
    return (chosen[..., None] * group_size + offsets).flatten(-2)
