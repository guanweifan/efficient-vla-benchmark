"""Public Hessian proxy: batch-normalized 2 X^T X, inverse-Cholesky weighting.

This is the released code's input-Hessian proxy, not the paper's exhaustive
action-space intervention. Layer normalization matters for global allocation.
"""

import torch
from torch import nn
from torch.nn import functional as F


class InputHessian:
    """Literal author recurrence, with inactive statistics offloaded to CPU.

    Batch examples, not tokens, normalize the covariance. Scaling inputs before
    GEMM preserves the reference rounding; summing raw Gram matrices and scaling
    once afterward is mathematically equal but not numerically identical.
    """

    def __init__(self, module):
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            raise ValueError("QVLA statistics require Linear or Conv2d")
        self.module = module
        self.columns = module.weight.flatten(1).shape[1]
        self.matrix = torch.zeros(self.columns, self.columns, dtype=torch.float32)
        self.examples = 0

    @torch.no_grad()
    def add_batch(self, inputs):
        x = inputs.detach()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        examples = x.shape[0]
        if not examples:
            raise ValueError("Empty QVLA calibration batch")
        if isinstance(self.module, nn.Conv2d):
            m = self.module
            x = F.unfold(x, m.kernel_size, m.dilation, m.padding, m.stride)
            x = x.permute(1, 0, 2).flatten(1)
        else:
            if x.ndim != 3:
                raise ValueError("Expected batched linear calibration inputs")
            x = x.reshape(-1, x.shape[-1]).T
        if x.shape[0] != self.columns:
            raise ValueError("QVLA covariance width mismatch")
        self.matrix.mul_(self.examples / (self.examples + examples))
        self.examples += examples
        with torch.autocast(device_type=x.device.type, enabled=False):
            scaled = torch.sqrt(torch.tensor(2.0 / self.examples, device=x.device)) * x.float()
            self.matrix.add_((scaled @ scaled.T).cpu())


def inverse_cholesky_diagonal(matrix, percdamp):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.numel():
        raise ValueError("Invalid Hessian statistics")
    h = matrix.float().clone()
    diag = torch.arange(h.shape[0], device=h.device)
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    # torch.diag materializes contiguous values. A strided diagonal mean can
    # change the damping scalar and amplify through the Cholesky inverse.
    h[diag, diag] += percdamp * torch.diag(h).mean()
    factor = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(factor)
    upper = torch.linalg.cholesky(inverse, upper=True)
    return torch.diag(upper)


def channel_scores(weight, diagonal, bits):
    w = weight.detach().float().flatten(1)
    squared_diagonal = diagonal.float().square().clamp_min(1e-12)
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
            score = ((w - quant).square() / squared_diagonal).sum(1)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("Nonfinite QVLA sensitivity")
        scores[bit] = score.cpu().tolist()
    return scores
