"""Native DiT attention with explicit per-head ATM and scalar post-projection OHB."""

import torch
from torch import nn
from torch.nn import functional as F


class BalancedAttention(nn.Module):
    def __init__(self, original, entry=None, collector=None):
        super().__init__()
        if original.training:
            raise ValueError("QuantVLA attention requires evaluation mode")
        entry = entry or {"alpha": [1.0] * original.num_heads, "beta": 1.0}
        self.original = original
        self.collector = collector
        alpha = torch.tensor(entry["alpha"], device=original.qkv.weight.device, dtype=torch.float32)
        beta = torch.tensor(entry["beta"], device=alpha.device, dtype=torch.float32)
        if (
            alpha.shape != (original.num_heads,)
            or beta.ndim != 0
            or not bool(
                torch.isfinite(alpha).all() & (alpha > 0).all() & torch.isfinite(beta) & (beta > 0)
            )
        ):
            raise ValueError("Invalid ATM/OHB coefficients")
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

    def forward(self, x):
        m = self.original
        batch, tokens, width = x.shape
        q, k, v = (
            m.qkv(x)
            .reshape(batch, tokens, 3, m.num_heads, m.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q, k = m.q_norm(q), m.k_norm(k)
        if self.collector is not None:
            logits = q.float() @ k.float().transpose(-1, -2) * m.scale
            std = logits.var(dim=(-2, -1), unbiased=False).clamp_min(1e-12).sqrt().mean(0).cpu()
        q = q * self.alpha.to(q.dtype).view(1, -1, 1, 1)
        if m.fused_attn:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            y = m.attn_drop((q * m.scale @ k.transpose(-1, -2)).softmax(-1)) @ v
        out = m.proj_drop(m.proj(y.transpose(1, 2).reshape(batch, tokens, width)))
        if self.collector is not None:
            rms = (out.float().square().mean() + 1e-12).sqrt().cpu()
            self.collector(std, rms)
        return out * self.beta.to(out.dtype)
