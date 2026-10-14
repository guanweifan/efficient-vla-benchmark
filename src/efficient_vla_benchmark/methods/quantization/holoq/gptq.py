"""Block-local GPTQ equations from the pinned official solver, retaining every scale."""

import torch


def mse_scales(weight):
    base = (weight.abs().amax(1) / 7).clamp_min(1e-8)
    alpha = torch.tensor((0.5, 0.75, 1.0, 1.25, 1.5), device=weight.device, dtype=weight.dtype)
    candidates = base[:, None] / alpha[None, :]
    expanded = candidates[:, :, None]
    q = (weight[:, None, :] / expanded).round().clamp(-8, 7) * expanded
    best = (q - weight[:, None, :]).square().mean(2).argmin(1)
    return candidates[torch.arange(len(weight), device=weight.device), best]


@torch.no_grad()
def quantize(weight, inputs, block_size=128, damping=0.01):
    if weight.ndim != 2 or inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("GPTQ input geometry mismatch")
    w = weight.detach().float()
    x = inputs.to(w)
    codes = torch.empty_like(w, dtype=torch.int8)
    scales = []
    for start in range(0, w.shape[1], block_size):
        end = min(start + block_size, w.shape[1])
        block = w[:, start:end].clone()
        xb = x[:, start:end]
        h = xb.T @ xb / max(len(xb), 1)
        h = (h + h.T) * 0.5
        diag = h.diagonal().clone()
        dead = diag <= 0
        h.diagonal()[dead] = 1
        block[:, dead] = 0
        mean = diag[~dead].mean() if bool((~dead).any()) else diag.new_tensor(1)
        h.diagonal().add_(damping * mean.clamp_min(1e-8))
        chol, info = torch.linalg.cholesky_ex(h)
        inv = torch.cholesky_inverse(chol) if not bool(info.any()) else torch.linalg.pinv(h)
        scale = mse_scales(block)
        scales.append(scale.cpu())
        for col in range(end - start):
            denom = inv[col, col]
            denom = torch.where(torch.isfinite(denom) & (denom.abs() >= 1e-12), denom, 1)
            original = block[:, col]
            q = (original / scale).round().clamp(-8, 7)
            codes[:, start + col] = q.to(torch.int8)
            error = (original - q * scale) / denom
            if col + 1 < block.shape[1]:
                block[:, col + 1 :] -= error[:, None] * inv[col, col + 1 :][None, :]
    return codes.cpu(), torch.stack(scales, 1)


@torch.no_grad()
def rtn(weight):
    scale = weight.float().abs().amax(1).clamp_min(1e-8) / 7
    codes = (weight.float() / scale[:, None]).round().clamp(-8, 7).to(torch.int8)
    return codes.cpu(), scale.cpu()[:, None]
