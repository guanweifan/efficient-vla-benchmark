import pytest
import torch
from torch.nn import functional as F

from efficient_vla_benchmark.model_ops.quantization.native_fp32 import PackedFP32Linear


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group", [64, 256])
@torch.inference_mode()
def test_transient_decode_matches_native_gemm_exactly(bits, group):
    torch.manual_seed(16)
    n, k = 96, 256
    limit = 1 << (bits - 1)
    codes = torch.randint(-limit, limit, (n, k), device="cuda", dtype=torch.int8)
    scales = torch.rand(n, k // group, device="cuda") * 0.01 + 0.001
    step = torch.rand(k, device="cuda") * 0.03 + 0.01
    bias = torch.randn(n, device="cuda")
    op = PackedFP32Linear(
        codes,
        scales,
        bits=bits,
        group_size=group,
        bias=bias,
        activation_step=step,
        activation_bits=4,
        activation_qmin=-8,
    )
    inputs = torch.randn(2, 17, k, device="cuda")
    dense = (codes.float() * scales.repeat_interleave(group, 1)).contiguous()
    reference = F.linear((inputs / step).round().clamp(-8, 7) * step, dense, bias)
    assert torch.equal(op(inputs), reference)
    assert torch.equal(op(inputs), reference)
    assert op.kernel_invocations == 2
    assert op.packed_weight.numel() == n * k * bits // 8
    assert not any(t.is_floating_point() and t.shape == (n, k) for t in op.buffers())
    assert op.execution_metadata()["transient_full_weight_workspace"]
    with pytest.raises(ValueError):
        op(inputs.bfloat16())
