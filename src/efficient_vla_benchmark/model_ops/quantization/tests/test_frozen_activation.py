import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear


@pytest.mark.gpu
def test_explicit_activation_step_is_not_rescaled_and_uses_declared_integer_range():
    q = torch.ones((16, 32), dtype=torch.int8, device="cuda")
    scale = torch.ones(16, device="cuda")
    step = torch.ones(32, device="cuda")
    x = torch.full((2, 32), -200.0, device="cuda")
    low = PackedLinear(q, scale, bits=4, activation_step=step, activation_qmin=-128).cuda()
    old = PackedLinear(q, scale, bits=4, activation_amax=step * 127).cuda()
    torch.testing.assert_close(low(x), torch.full((2, 16), -4096.0, device="cuda"))
    torch.testing.assert_close(old(x), torch.full((2, 16), -4064.0, device="cuda"))
    with pytest.raises(ValueError):
        PackedLinear(q, scale, bits=4, activation_amax=step, activation_step=step)
