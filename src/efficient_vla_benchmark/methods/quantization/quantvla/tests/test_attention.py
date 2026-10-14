from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from efficient_vla_benchmark.methods.quantization.quantvla.implementations.cogact import attention


def native():
    return SimpleNamespace(
        training=False,
        qkv=nn.Linear(8, 24),
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
        proj=nn.Linear(8, 8),
        proj_drop=nn.Identity(),
        attn_drop=nn.Identity(),
        fused_attn=True,
        num_heads=2,
        head_dim=4,
        scale=0.5,
    )


def test_attention_matches_native_and_author_scale_placement():
    torch.manual_seed(9)
    model = native()
    x = torch.randn(1, 8, 8)
    q, k, v = model.qkv(x).reshape(1, 8, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    expected = model.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 8, 8))
    torch.testing.assert_close(attention.BalancedAttention(model)(x), expected, rtol=0, atol=0)
    alpha = [1.3, 0.7]
    beta = 0.8
    expected = (
        model.proj(
            F.scaled_dot_product_attention(q * torch.tensor(alpha)[None, :, None, None], k, v)
            .transpose(1, 2)
            .reshape(1, 8, 8)
        )
        * beta
    )
    torch.testing.assert_close(
        attention.BalancedAttention(model, dict(alpha=alpha, beta=beta))(x), expected
    )
    model.fused_attn = False
    torch.testing.assert_close(
        attention.BalancedAttention(model, dict(alpha=alpha, beta=beta))(x), expected
    )
    with pytest.raises(ValueError, match="Invalid"):
        attention.BalancedAttention(model, dict(alpha=[0, 1], beta=1))
