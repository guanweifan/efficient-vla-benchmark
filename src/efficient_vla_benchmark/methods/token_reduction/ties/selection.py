"""Budget allocation to attention top-K and uniform remaining-token sampling."""

import math

import torch


def select_visual(scores, keep_ratio, fraction, generator):
    if scores.ndim != 1 or not len(scores) or not bool(torch.isfinite(scores).all()):
        raise ValueError("TIES requires finite visual attention scores")
    if not 0 < keep_ratio <= 1 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid TIES allocation")
    count = max(1, int(len(scores) * keep_ratio))
    attention_count = int(fraction * count)
    order = scores.argsort(descending=True, stable=True)
    selected = torch.zeros(len(scores), device=scores.device, dtype=torch.bool)
    selected[order[:attention_count]] = True
    remaining = (~selected).nonzero().flatten()
    uniform_count = count - attention_count
    if uniform_count:
        indices = torch.randperm(len(remaining), generator=generator, device=scores.device)
        selected[remaining[indices[:uniform_count]]] = True
    return selected, dict(attention_tokens=attention_count, uniform_tokens=uniform_count)
