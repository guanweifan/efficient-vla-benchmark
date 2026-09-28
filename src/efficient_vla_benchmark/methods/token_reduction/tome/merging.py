"""Independent implementation of ToMe matching and token-mass equations."""

import torch
from torch.nn import functional as F


def bipartite_groups(keys, removal, prefix):
    """Return old-to-new membership in the released unmerged-A/B output order.

    Protect all native prefix tokens (CLS and DINO registers). For zero/one
    prefix the ordering equals the released implementation; additional prefixes
    stay at the front in native order and never participate in a merge edge.
    """
    if keys.ndim != 3 or type(prefix) is not int or not 0 <= prefix < keys.shape[1]:
        raise ValueError("Expected batched key vectors with a valid native prefix")
    if type(removal) is not int or removal < 0:
        raise ValueError("removal must be a nonnegative integer")
    batch, length, _ = keys.shape
    r = min(removal, (length - prefix) // 2)
    identity = torch.arange(length, device=keys.device).expand(batch, -1)
    if r == 0:
        return identity, length
    normalized = F.normalize(keys.float(), dim=-1)
    left = torch.arange(0, length, 2, device=keys.device)
    right = torch.arange(1, length, 2, device=keys.device)
    left, right = left[left >= prefix], right[right >= prefix]
    similarities = normalized[:, left] @ normalized[:, right].transpose(-2, -1)
    confidence, destination = similarities.max(-1)
    order = torch.argsort(confidence, dim=-1, descending=True, stable=True)
    sources = left[order[:, :r]]
    remaining = left[order[:, r:]]
    if prefix:
        remaining = remaining.sort(-1).values
    targets = right[destination.gather(1, order[:, :r])]
    protected = torch.arange(prefix, device=keys.device).expand(batch, -1)
    survivors = torch.cat((protected, remaining, right.expand(batch, -1)), dim=1)
    groups = torch.empty_like(identity)
    groups.scatter_(1, survivors, torch.arange(length - r, device=keys.device).expand(batch, -1))
    groups.scatter_(1, sources, groups.gather(1, targets))
    return groups, length - r


def weighted_merge(features, mass, groups, count):
    if features.ndim != 3 or mass.shape != features.shape[:2] or groups.shape != mass.shape:
        raise ValueError("Features, token masses and memberships must align")
    batch, _, channels = features.shape
    summed_mass = mass.new_zeros((batch, count)).scatter_add_(1, groups, mass)
    weighted = features.float() * mass[..., None]
    merged = weighted.new_zeros((batch, count, channels))
    merged.scatter_add_(1, groups[..., None].expand_as(features), weighted)
    return (merged / summed_mass[..., None]).to(features.dtype), summed_mass


def attention_with_mass(attention, hidden, mass):
    """Native timm projections with ToMe Eq. 1; return mean-head keys as metric."""
    batch, length, channels = hidden.shape
    projected = attention.qkv(hidden).reshape(
        batch, length, 3, attention.num_heads, attention.head_dim
    )
    q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
    metric = k.mean(1)
    q, k = attention.q_norm(q), attention.k_norm(k)
    bias = mass.log()[:, None, None].to(q.dtype)
    if attention.fused_attn:
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
    else:
        logits = (q * attention.scale) @ k.transpose(-2, -1)
        result = attention.attn_drop((logits + bias).softmax(-1)) @ v
    result = result.transpose(1, 2).reshape(batch, length, channels)
    return attention.proj_drop(attention.proj(result)), metric
