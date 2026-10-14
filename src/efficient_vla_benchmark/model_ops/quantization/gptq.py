"""Full-Hessian GPTQ with global cross-block compensation.

Independent implementation of Frantar et al., Algorithm 1 and the public
IST-DASLab/gptq solver (fasterquant). No activation quantization or paper policy.
The symmetric ungrouped grid matches the public quantizer, including its
asymmetric signed endpoint and zero-column handling.
"""

import torch


@torch.inference_mode()
def symmetric_scales(weight, bits):
    maximum = (1 << bits) - 1
    xmin = torch.minimum(weight.amin(1), torch.zeros_like(weight[:, 0]))
    xmax = torch.maximum(weight.amax(1), torch.zeros_like(weight[:, 0]))
    xmax = torch.maximum(xmin.abs(), xmax)
    xmin = torch.where(xmin < 0, -xmax, xmin)
    dead = (xmin == 0) & (xmax == 0)
    xmin = torch.where(dead, -torch.ones_like(xmin), xmin)
    xmax = torch.where(dead, torch.ones_like(xmax), xmax)
    return (xmax - xmin) / maximum


@torch.inference_mode()
def quantize(weight, hessian, bits, block_size=128, damp_percent=0.01):
    if bits not in (4, 8) or type(bits) is not int:
        raise ValueError("GPTQ precision must be 4 or 8")
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Invalid GPTQ geometry")
    if not bool(torch.isfinite(weight).all() & torch.isfinite(hessian).all()):
        raise ValueError("Nonfinite GPTQ input")
    if block_size < 1 or not 0 < damp_percent <= 1:
        raise ValueError("Invalid GPTQ numerical settings")
    w = weight.detach().float().clone()
    h = hessian.detach().float().clone()
    scale = symmetric_scales(w, bits)
    dead = h.diagonal() == 0
    h.diagonal()[dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damp_percent * h.diagonal().mean())
    # Upper Cholesky of the inverse, NOT the inverse itself.
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = torch.empty_like(w, dtype=torch.int8)
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for start in range(0, w.shape[1], block_size):
        end = min(start + block_size, w.shape[1])
        block = w[:, start:end].clone()
        errors = torch.empty_like(block)
        for local in range(end - start):
            col = block[:, local]
            code = (col / scale).round().clamp(qmin, qmax)
            quantized = code * scale
            error = (col - quantized) / upper[start + local, start + local]
            block[:, local:] -= error[:, None] * upper[start + local, start + local : end][None, :]
            errors[:, local] = error
            codes[:, start + local] = code.to(torch.int8)
        if end < w.shape[1]:
            w[:, end:] -= errors @ upper[start:end, end:]
    return codes, scale
