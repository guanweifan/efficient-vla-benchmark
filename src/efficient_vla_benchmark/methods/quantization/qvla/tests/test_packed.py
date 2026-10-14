import pytest
import torch
from torch import nn

from efficient_vla_benchmark.methods.quantization.qvla.allocation import greedy_allocate


def test_absolute_cost_not_marginal_cost():
    scores = {"a": {16: [0.0, 0.0], 8: [1.0, 1.6], 4: [1.1, 2.0], 0: [50.0, 50.0]}}
    gates, stats = greedy_allocate(scores, (0, 4, 8, 16), 10)
    assert gates == {"a": [8, 8]}
    assert stats["final_avg_bits"] == 8


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("convolution", [False, True])
@torch.inference_mode()
def test_mixed_all_bits_preserves_rows_and_bias(dtype, convolution):
    from efficient_vla_benchmark.methods.quantization.qvla.packed import (
        MixedBitConv2d,
        MixedBitLinear,
    )

    torch.manual_seed(5)
    native = (
        (nn.Conv2d(3, 15, 3, stride=2, padding=1) if convolution else nn.Linear(97, 15))
        .cuda()
        .to(dtype)
    )
    gates = torch.tensor([0, 2, 4, 8, 16] * 3)
    original = native.weight.clone()
    runtime = MixedBitConv2d(native, gates) if convolution else MixedBitLinear(native, gates)
    assert torch.equal(native.weight, original)
    for i, bit in enumerate(gates.tolist()):
        w = native.weight[i].float()
        if bit == 0:
            native.weight[i].zero_()
        elif bit < 16:
            maximum = (1 << (bit - 1)) - 1
            scale = w.abs().max().clamp_min(1e-8) / maximum
            native.weight[i].copy_((w / scale).round().clamp(-maximum - 1, maximum) * scale)
    x = torch.randn((2, 3, 18, 17) if convolution else (2, 17, 97), device="cuda", dtype=dtype)
    actual, expected = runtime(x), native(x)
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.02 if dtype == torch.bfloat16 else 0.002,
        rtol=0.02 if dtype == torch.bfloat16 else 0.002,
    )
    zero = actual[:, 0] if convolution else actual[..., 0]
    torch.testing.assert_close(zero, torch.ones_like(zero) * native.bias[0], atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [0, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_extreme_uniform_scopes_and_reload(bits, dtype):
    from efficient_vla_benchmark.methods.quantization.qvla.packed import MixedBitLinear

    torch.manual_seed(7)
    native = nn.Linear(128, 32).cuda().to(dtype)
    gates = torch.full((32,), bits)
    first = MixedBitLinear(native, gates)
    second = MixedBitLinear(native, gates)
    second.load_state_dict(first.state_dict())
    x = torch.randn(3, 128, device="cuda", dtype=dtype)
    actual = first(x)
    expected = native(x) if bits == 16 else native.bias.expand(3, 32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(second(x), actual, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [2, 4, 8, 16])
@torch.inference_mode()
def test_uniform_groups_bypass_scatter_without_changing_results(bits):
    from efficient_vla_benchmark.methods.quantization.qvla.packed import MixedBitLinear

    torch.manual_seed(11)
    native = nn.Linear(97, 65).cuda().bfloat16()
    runtime = MixedBitLinear(native, torch.full((65,), bits))
    x = torch.randn(33, 97, device="cuda", dtype=torch.bfloat16)
    assert runtime.uniform_group == str(bits)
    torch.testing.assert_close(runtime(x), runtime.groups[str(bits)](x), atol=0, rtol=0)
    mixed = MixedBitLinear(native, torch.tensor([0] + [bits] * 64))
    assert mixed.uniform_group is None
    torch.testing.assert_close(mixed(x)[..., 0], native.bias[0].expand(33), atol=0, rtol=0)
