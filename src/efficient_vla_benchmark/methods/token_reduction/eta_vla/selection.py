"""ILSA equations 3--8; semantic probes do not replace native attention."""

import torch
import torch.nn.functional as F


def text_anchors(visual, text):
    if visual.ndim != 2 or text.ndim != 2 or visual.shape[1] != text.shape[1]:
        raise ValueError("ILSA needs aligned native hidden features")
    if not len(visual) or not len(text):
        raise ValueError("ILSA needs visual and language rows")
    with torch.autocast(device_type=visual.device.type, enabled=False):
        alignment = (visual.float() @ text.float().T).softmax(dim=-1).mean(dim=0)
        return (alignment > alignment.mean()).nonzero().flatten()


def semantic_scores(queries, keys, visual_ids):
    """Q=[heads,raters,dim], K=[heads,ALL current tokens,dim], no RoPE/mask."""
    if (
        queries.ndim != 3
        or keys.ndim != 3
        or queries.shape[0] != keys.shape[0]
        or queries.shape[-1] != keys.shape[-1]
        or queries.shape[1] == 0
    ):
        raise ValueError("Invalid ILSA semantic probe geometry")
    with torch.autocast(device_type=queries.device.type, enabled=False):
        logits = queries.float() @ keys.float().transpose(-2, -1) / queries.shape[-1] ** 0.5
        return logits.softmax(dim=-1).mean(dim=(0, 1))[visual_ids]


def view_diversity(features, view_ids):
    """Equation 7 against ALL current visual members of the corresponding view."""
    if features.ndim != 2 or view_ids.shape != features.shape[:1] or not len(features):
        raise ValueError("Invalid ILSA view geometry")
    with torch.autocast(device_type=features.device.type, enabled=False):
        unit = F.normalize(features.float(), dim=-1, eps=1e-12)
        result = torch.zeros(len(features), device=features.device, dtype=torch.float32)
        for view in view_ids.unique():
            ids = (view_ids == view).nonzero().flatten()
            if len(ids) < 2:
                continue
            rows = unit[ids]
            # Sum_j cosine(i,j), excluding the true diagonal even for a zero vector.
            similarities = rows @ rows.sum(dim=0)
            diagonal = rows.square().sum(dim=-1)
            result[ids] = 1 - (similarities - diagonal) / (len(ids) - 1)
        return result


def select_tokens(features, scores, view_ids, keep_ratio, recycle_ratio):
    count = len(features)
    if count < 2 or scores.shape != (count,) or not bool(torch.isfinite(scores).all()):
        raise ValueError("ILSA requires finite visual scores and at least two tokens")
    budget = min(count, max(2, int(count * keep_ratio)))
    recycled = min(budget - 1, max(1, int(budget * recycle_ratio)))
    semantic = budget - recycled
    order = scores.argsort(descending=True, stable=True)
    chosen = torch.zeros(count, device=features.device, dtype=torch.bool)
    chosen[order[:semantic]] = True
    diversity = view_diversity(features, view_ids)
    remaining = (~chosen).nonzero().flatten()
    ranked = diversity[remaining].argsort(descending=True, stable=True)
    chosen[remaining[ranked[:recycled]]] = True
    return chosen, dict(
        input_tokens=count, output_tokens=budget, semantic_tokens=semantic, recycled_tokens=recycled
    )
