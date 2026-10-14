"""Public Hessian proxy: batch-normalized 2 X^T X, inverse-Cholesky weighting.

This is the released code's input-Hessian proxy, not the paper's exhaustive
action-space intervention. Layer normalization matters for global allocation.
"""

import torch


def hessian_scale(sum_gram, examples, percdamp):
    if examples <= 0 or sum_gram.ndim != 2 or sum_gram.shape[0] != sum_gram.shape[1]:
        raise ValueError("Invalid Hessian statistics")
    h = sum_gram.float() * (2.0 / examples)
    diag = torch.arange(h.shape[0], device=h.device)
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    h[diag, diag] += percdamp * h.diagonal().mean()
    factor = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(factor)
    upper = torch.linalg.cholesky(inverse, upper=True)
    return upper.diagonal().square().clamp_min(1e-12).reciprocal()


def channel_scores(weight, scale, bits):
    w = weight.detach().float().flatten(1)
    scores = {}
    for bit in bits:
        if bit == 16:
            score = torch.zeros(w.shape[0], device=w.device)
        else:
            if bit == 0:
                quant = torch.zeros_like(w)
            else:
                maximum = (1 << (bit - 1)) - 1
                step = w.abs().amax(1, keepdim=True).clamp_min(1e-6) / maximum
                quant = (w / step).round().clamp(-maximum - 1, maximum) * step
            score = ((w - quant).square() * scale).sum(1)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("Nonfinite QVLA sensitivity")
        scores[bit] = score.cpu().tolist()
    return scores
