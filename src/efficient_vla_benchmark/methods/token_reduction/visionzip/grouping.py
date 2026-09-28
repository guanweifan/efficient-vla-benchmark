"""Received attention, uniform residual targets and released contextual aggregation."""

import torch
import torch.nn.functional as F


def visual_evidence(query, key, scale):
    if query.ndim != 4 or query.shape != key.shape or query.shape[0] != 1:
        raise ValueError("VisionZip requires aligned native SigLIP Q/K")
    with torch.autocast(device_type=query.device.type, enabled=False):
        weights = ((query.float() * scale) @ key.float().transpose(-1, -2)).softmax(-1)
        importance = weights.mean(dim=(1, 2))[0]
        metric = key.float().mean(dim=1)[0]
    return importance, metric


def compress_view(features, importance, metric, target, dominant_ratio):
    if (
        features.ndim != 2
        or metric.ndim != 2
        or importance.shape != (features.shape[0],)
        or metric.shape[0] != features.shape[0]
        or not features.device == importance.device == metric.device
        or not all(x.is_floating_point() for x in (features, importance, metric))
        or not bool(
            torch.isfinite(features).all()
            & torch.isfinite(importance).all()
            & torch.isfinite(metric).all()
        )
    ):
        raise ValueError("VisionZip requires finite aligned patch features and visual evidence")
    count = features.shape[0]
    if type(target) is not int or not 2 <= target <= count:
        raise ValueError("VisionZip target must retain at least two and at most all patches")
    if type(dominant_ratio) not in (int, float) or not 0 < dominant_ratio < 1:
        raise ValueError("Both dominant and contextual contributions must remain active")
    dominant_count = min(target - 1, max(1, int(target * dominant_ratio)))
    contextual_count = target - dominant_count
    dominant = (
        importance.float().argsort(descending=True, stable=True)[:dominant_count].sort().values
    )
    residual_mask = torch.ones(count, dtype=torch.bool, device=features.device)
    residual_mask[dominant] = False
    residual = residual_mask.nonzero().flatten()
    # The official target stride uses residual ORIGINAL position order, not saliency order.
    stride = max(1, residual.numel() // contextual_count)
    target_rows = torch.arange(0, residual.numel(), stride, device=features.device)[
        :contextual_count
    ]
    seeds = residual[target_rows]
    merging_mask = torch.ones(residual.numel(), dtype=torch.bool, device=features.device)
    merging_mask[target_rows] = False
    merging = residual[merging_mask]
    with torch.autocast(device_type=features.device.type, enabled=False):
        normalized = F.normalize(metric.float(), dim=-1, eps=1e-12)
        similarities = normalized[merging] @ normalized[seeds].T
        assignment = similarities.argmax(dim=-1)
        membership = F.one_hot(assignment, contextual_count).float()
        counts = membership.sum(0)
        aggregate = (membership.T @ features[merging].float()) / counts.clamp_min(1)[:, None]
        # Released code is target + mean(other assigned members), not a pooled mean.
        contextual = features[seeds].float() + aggregate
        compressed = torch.cat((features[dominant].float(), contextual))
    anchors = seeds.clone()
    anchors.scatter_reduce_(0, assignment, merging, reduce="amax", include_self=True)
    all_anchors = torch.cat((dominant, anchors))
    order = all_anchors.argsort(stable=True)
    return compressed[order].to(features.dtype), all_anchors[order]
