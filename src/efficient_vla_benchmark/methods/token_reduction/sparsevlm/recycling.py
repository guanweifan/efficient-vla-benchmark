"""Equations 6-11 of SparseVLM v1 with native-position-safe merge anchors."""

import torch
from torch.nn import functional as F


def text_raters(visual, text):
    if not len(visual) or not len(text):
        raise ValueError("SparseVLM requires both projected visual and language tokens")
    relevance = (visual.float() @ text.float().T).softmax(-1).mean(0)
    return relevance >= relevance.mean()


def density_clusters(features, count, neighbors):
    n = len(features)
    if not 1 <= count <= n:
        raise ValueError("Invalid cluster count")
    if n == 1:
        return torch.zeros(1, dtype=torch.long, device=features.device)
    values = features.float()
    squared = torch.cdist(values, values).square()
    nearest = squared.clone()
    nearest.fill_diagonal_(float("inf"))
    k = min(neighbors, n - 1)
    # Work in log density: mathematically equivalent scores without exp underflow.
    log_density = -nearest.topk(k, largest=False).values.mean(-1)
    higher = log_density[None, :] > log_density[:, None]
    distances = squared.sqrt()
    delta = torch.where(
        higher.any(-1),
        distances.masked_fill(~higher, float("inf")).amin(-1),
        distances.amax(-1),
    )
    score = log_density + delta.clamp_min(torch.finfo(torch.float32).tiny).log()
    centers = torch.argsort(score, descending=True, stable=True)[:count]
    similarity = F.normalize(values, dim=-1) @ F.normalize(values[centers], dim=-1).T
    assignment = similarity.argmax(-1)
    # Every center owns itself, including coincident-feature tie cases.
    assignment[centers] = torch.arange(count, device=features.device)
    return assignment


def prune_and_recycle(features, probabilities, positions, config):
    """Return selected local visual indices, replacements and actual stage stats."""
    n = len(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != n or not len(probabilities):
        raise ValueError("Expected rater-to-current-visual probabilities")
    rank = int(torch.linalg.matrix_rank(probabilities.float(), rtol=config.rank_rtol))
    removed_count = min(n - 1, int(config.rank_scale * (n - rank)))
    if removed_count <= 0:
        return (
            torch.arange(n, device=features.device),
            features,
            {
                "rank": rank,
                "removed": 0,
                "recycled": 0,
                "clusters": 0,
            },
        )
    ordering = torch.argsort(probabilities.mean(0), descending=True, stable=True)
    kept = ordering[: n - removed_count]
    recycle_count = min(removed_count, max(1, int(removed_count * config.recycle_ratio)))
    recycled = ordering[n - removed_count : n - removed_count + recycle_count]
    cluster_count = min(recycle_count, max(1, int(recycle_count * config.cluster_ratio)))
    assignment = density_clusters(features[recycled], cluster_count, config.neighbors)
    anchors, replacements = [], []
    for cluster in range(cluster_count):
        members = recycled[assignment == cluster]
        # Causal-safe anchor: every member is at or before the group's position.
        anchor = members[positions[members].argmax()]
        anchors.append(anchor)
        # Eq. 11 is a SUM, deliberately not the released fixed-budget mean variant.
        replacements.append(features[members].float().sum(0).to(features.dtype))
    anchors = torch.stack(anchors)
    ids = torch.cat((kept, anchors))
    values = torch.cat((features[kept], torch.stack(replacements)))
    order = positions[ids].argsort()
    return (
        ids[order],
        values[order],
        {
            "rank": rank,
            "removed": removed_count,
            "recycled": recycle_count,
            "clusters": cluster_count,
        },
    )
