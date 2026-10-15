import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.integer_linear import IntegerLinear


def test_integer_contract_and_rejections():
    codes = torch.zeros(3, 7, dtype=torch.int8)
    scales = torch.ones(3)
    op = IntegerLinear(codes, scales)
    assert op.execution_metadata()["integer_mma"] is True
    assert op.execution_metadata()["activation_scale_axis"] == "token_dynamic"
    assert not list(op.parameters())
    for bits in (2, 16, True):
        with pytest.raises(ValueError):
            IntegerLinear(codes, scales, bits=bits)
    with pytest.raises(ValueError, match="per output channel"):
        IntegerLinear(codes, torch.ones(7))
    with pytest.raises(ValueError, match="finite and positive"):
        IntegerLinear(codes, torch.zeros(3))
    with pytest.raises(ValueError, match="no CPU fallback"):
        op(torch.ones(2, 7))


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,width,outputs", [(1, 97, 41), (35, 129, 67), (280, 1024, 128)])
@torch.inference_mode()
def test_integer_reference(bits, dtype, rows, width, outputs):
    torch.manual_seed(19)
    bound = 1 << (bits - 1)
    w = torch.randint(-bound, bound, (outputs, width), dtype=torch.int8, device="cuda")
    ws = torch.rand(outputs, device="cuda") * 0.01 + 0.001
    bias = torch.randn(outputs, device="cuda") * 0.1
    op = IntegerLinear(w, ws, bits=bits, bias=bias)
    # Deliberately strided input, zero token and multiple leading dimensions.
    x = torch.randn(1, rows, width * 2, device="cuda", dtype=dtype)[..., ::2]
    x[:, 0] = 0
    xf = x.reshape(rows, width).float()
    xs = xf.abs().amax(1).clamp_min(1e-5) / 127
    xq = (xf / xs[:, None]).round().clamp(-127, 127)
    # FP64 products are exact for integer sums of these sizes, independent of
    # both the Triton implementation and CUDA's reduced-precision float GEMMs.
    integer_sum = (xq.double() @ w.double().T).float()
    expected = (integer_sum * xs[:, None] * ws + bias).to(dtype).reshape(1, rows, outputs)
    actual = op(x)
    tolerance = {torch.float16: 0.002, torch.bfloat16: 0.016, torch.float32: 2e-5}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert op.kernel_invocations == 1
    assert op(x[:, :0]).shape == (1, 0, outputs)
    assert not any(t.is_floating_point() and t.shape == w.shape for t in op.buffers())
    restored = IntegerLinear(torch.zeros_like(w), torch.ones_like(ws), bits=bits, bias=bias)
    restored.load_state_dict(op.state_dict())
    torch.testing.assert_close(restored(x), actual, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_quantized_codes_exactly_match_reference(dtype):
    from efficient_vla_benchmark.model_ops.quantization.integer_kernel import quantize_tokens

    torch.manual_seed(71)
    x = torch.randn(593, 1024, device="cuda", dtype=dtype)
    x[0] = 0
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(593, device="cuda", dtype=torch.float32)
    quantize_tokens[(593,)](x, q, scales, 593, 1024, 1e-5, 1024)
    reference_scales = x.float().abs().amax(1).clamp_min(1e-5) / 127
    expected = (x.float() / reference_scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(scales, reference_scales, atol=0, rtol=0)
    torch.testing.assert_close(q, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [4, 8])
@torch.inference_mode()
def test_quantizer_rounding_and_compiled_integer_mma(bits):
    import triton

    from efficient_vla_benchmark.model_ops.quantization.integer_kernel import (
        integer_gemm,
        quantize_tokens,
    )

    x = torch.zeros(1, 128, device="cuda")
    x[0, :9] = torch.tensor([127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0], device="cuda")
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(1, device="cuda")
    quantize_tokens[(1,)](x, q, s, 1, 128, 1e-5, 128)
    torch.testing.assert_close(q, x.round().to(torch.int8), rtol=0, atol=0)
    w = torch.ones(32, 128, device="cuda", dtype=torch.int8)
    op = IntegerLinear(w, torch.ones(32, device="cuda"), bits=bits)
    output = torch.empty(1, 32, device="cuda")
    compiled = integer_gemm[(1, 1)](
        q,
        op.packed_weight,
        s,
        op.weight_scale,
        op.weight_scale,
        output,
        1,
        32,
        128,
        bits,
        False,
        32,
        64,
        64,
        num_warps=4,
        num_stages=3,
    )
    torch.testing.assert_close(output, q.float().sum(1, keepdim=True).expand_as(output))
    ptx = compiled.asm["ptx"]
    assert "mma.sync" in ptx and ".s32.s8.s8.s32" in ptx
    assert triton is not None
