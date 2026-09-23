"""Paper-based max-min visual dispersion, with explicit duplicate exclusion."""

import torch
from torch.nn import functional as F


def max_min_selection(features, count):
    if features.ndim != 2 or type(count) is not int or not 1 <= count <= len(features):
        raise ValueError("Expected visual feature rows and a valid retained count")
    n = len(features)
    if count == n:
        return torch.arange(n, device=features.device)
    vectors = F.normalize(features.float(), dim=-1)
    distance = (1 - vectors @ vectors.T).clamp_min(0)
    # The farthest nearest-neighbour is the first seed. Exclude the self edge.
    initial = distance.clone()
    initial.fill_diagonal_(float("inf"))
    scores = initial.amin(0)
    available = torch.ones(n, dtype=torch.bool, device=features.device)
    selected = torch.empty(count, dtype=torch.long, device=features.device)
    nearest = torch.full_like(scores, float("inf"))
    for index in range(count):
        chosen = scores.masked_fill(~available, -float("inf")).argmax()
        selected[index] = chosen
        available.scatter_(0, chosen.reshape(1), False)
        nearest = torch.minimum(nearest, distance.index_select(0, chosen.reshape(1))[0])
        scores = nearest
    return selected.sort().values


def visual_selection(hidden, visual, ratio):
    ids = visual.nonzero().flatten()
    chosen = max_min_selection(hidden[ids], max(1, round(len(ids) * ratio)))
    keep = ~visual
    keep[ids[chosen]] = True
    return keep
