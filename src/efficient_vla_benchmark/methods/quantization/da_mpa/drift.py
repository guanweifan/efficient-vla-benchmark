"""DA-PTQ Eq. 12-19 and explicit native action normalization."""

import numpy as np
import torch


def structural_jacobian(actions, damping):
    q = torch.as_tensor(actions, dtype=torch.float64)
    if q.ndim != 2 or q.shape[1] != 7 or not torch.isfinite(q).all():
        raise ValueError("Drift calibration needs finite seven-dimensional actions")
    theta = q.cumsum(-1)

    def suffix(x):
        return x.flip(-1).cumsum(-1).flip(-1)

    j = torch.stack((-suffix(theta.sin()), suffix(theta.cos()), torch.ones_like(theta)), dim=1)
    inverse = j.transpose(1, 2) @ torch.linalg.inv(j @ j.transpose(1, 2) + damping * torch.eye(3))
    return j, inverse


def drift_weights(actions, config):
    _, inverse = structural_jacobian(actions, config.damping)
    axis = torch.tensor(
        [config.translation_weight, config.translation_weight, config.rotation_weight],
        dtype=inverse.dtype,
    )
    score = (inverse.abs() * axis).sum(-1).mean(0)
    return (score / score.mean().clamp_min(1e-12)).float()


def normalize_actions(raw, stats):
    x = np.asarray(raw, dtype=np.float32).copy()
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    if x.ndim != 2 or x.shape[1] != 7 or mask.shape != (7,) or mask[6] or not np.isfinite(x).all():
        raise ValueError("Invalid native action calibration convention")
    x[:, 6] = 1 - np.clip(x[:, 6], 0, 1)
    x = np.where(mask, np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1), x)
    if "min" in stats and "max" in stats:
        x = np.where(np.asarray(stats["min"]) == np.asarray(stats["max"]), 0, x)
    return x.astype(np.float32)


def retained_blocks(scores, blocks, ratio, excluded):
    blocks = sorted(set(blocks))
    if not blocks or any(not np.isfinite(s) or s < 0 for s in scores.values()):
        raise ValueError("Invalid drift sensitivity scores")
    eligible = blocks[:-excluded] if excluded else blocks
    ranked = sorted(eligible, key=lambda b: (-scores[b], b))
    count = max(1, round(len(eligible) * ratio))
    if count >= len(eligible):
        raise ValueError("DA-MPA must leave at least one eligible DiT block quantized")
    return set(ranked[:count]) | set(blocks[len(eligible) :])
