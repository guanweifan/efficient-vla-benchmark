import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 16, 605])
@torch.inference_mode()
def test_real_awq_gemm_and_gemv_match_dequantized_linear(dtype, rows):
    from ..implementations.openvla_oft.vendor import (
        reference_module,
    )

    torch.manual_seed(5)
    WQLinear = reference_module("quantize.qmodule").WQLinear
    quantize = reference_module("quantize.quantizer").pseudo_quantize_tensor
    linear = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=dtype)
    dequantized, scales, zeros = quantize(
        linear.weight, n_bit=4, zero_point=True, q_group_size=128, get_scale_zp=True
    )
    linear.weight.copy_(dequantized)
    packed = WQLinear.from_linear(linear, 4, 128, False, scales, zeros)
    inputs = torch.randn(1, rows, 256, device="cuda", dtype=dtype)
    expected = linear(inputs).float()
    actual = packed(inputs).float()
    assert packed.qweight.dtype == torch.int16
    assert packed.qweight.numel() * 4 == linear.weight.numel()
    assert torch.isfinite(actual).all()
    relative_error = (actual - expected).norm() / expected.norm()
    assert relative_error < 0.025
    torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.035)
    restored = WQLinear.from_linear(linear, 4, 128, init_only=True)
    restored.load_state_dict(packed.state_dict(), strict=True)
    torch.testing.assert_close(restored(inputs), packed(inputs), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 16, 605])
@torch.inference_mode()
def test_explicit_fp16_kernel_matches_author_with_casts(rows):
    from ..implementations.openvla_oft.runtime import KernelPrecisionLinear
    from ..implementations.openvla_oft.vendor import reference_module

    torch.manual_seed(29)
    WQLinear = reference_module("quantize.qmodule").WQLinear
    quantize = reference_module("quantize.quantizer").pseudo_quantize_tensor
    linear = torch.nn.Linear(256, 256, bias=True, device="cuda", dtype=torch.bfloat16)
    weights, scales, zeros = quantize(
        linear.weight, n_bit=4, zero_point=True, q_group_size=128, get_scale_zp=True
    )
    linear.weight.copy_(weights)
    packed = WQLinear.from_linear(linear, 4, 128, False, scales, zeros)
    original = packed.qweight.clone()
    x = torch.randn(1, rows, 256, device="cuda", dtype=torch.bfloat16)
    author = WQLinear.from_linear(linear, 4, 128, init_only=True).half()
    author.load_state_dict(packed.state_dict(), strict=True)
    expected = author(x.half()).bfloat16()
    actual = KernelPrecisionLinear(packed, torch.float16)(x)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(packed.qweight, original, atol=0, rtol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.gpu
def test_kernel_conversion_rejects_buffer_overflow():
    from ..implementations.openvla_oft.runtime import KernelPrecisionLinear
    from ..implementations.openvla_oft.vendor import reference_module

    packed = reference_module("quantize.qmodule").WQLinear(
        4, 128, 256, 256, False, "cuda", dtype=torch.bfloat16
    )
    packed.scales.fill_(1e10)
    with pytest.raises(ValueError, match="overflow"):
        KernelPrecisionLinear(packed, torch.float16)
