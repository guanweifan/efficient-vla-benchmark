"""Released semantic-mass and Canny-edge selection shared across bindings."""

import cv2
import numpy as np
import torch


def attention_mass(z, threshold, max_prune):
    # Follow executable reference, not its contradictory high/low-speed comments.
    return 1.0 if z < threshold else 1 - (min(z, 1) - threshold) * max_prune / (1 - threshold)


def edge_tokens(image, low, high):
    rgb = np.asarray(image)
    if rgb.shape != (224, 224, 3):
        raise ValueError("SP-VLA requires policy-preprocessed 224x224 images")
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), low, high)
    return edges.reshape(16, 14, 16, 14).any((1, 3)).flatten()


def semantic_selection(q, k, mass):
    attention = ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5).softmax(-1)
    score = attention.mean(1).mean(1)[0]
    order = torch.argsort(score, descending=True, stable=True)
    selected = torch.zeros_like(score, dtype=torch.bool)
    if mass >= 1:
        return ~selected
    count = min(score.numel(), int((score[order].cumsum(0) < mass).sum().item()) + 1)
    selected[order[:count]] = True
    return selected
