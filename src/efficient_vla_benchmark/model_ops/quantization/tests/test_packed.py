import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.packed import pack_signed, unpack_signed


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("width", [1, 17, 64, 127])
def test_signed_roundtrip(bits, width):
    bound = 1 << (bits - 1)
    q = torch.randint(-bound, bound, (7, width), dtype=torch.int8)
    p = pack_signed(q, bits)
    assert p.numel() == 7 * ((width + 8 // bits - 1) // (8 // bits))
    assert torch.equal(unpack_signed(p, bits, width), q)


def test_bad_inputs():
    with pytest.raises(ValueError):
        pack_signed(torch.ones(2, 3), 4)
    with pytest.raises(ValueError):
        pack_signed(torch.full((2, 3), 2, dtype=torch.int8), 2)
    with pytest.raises(ValueError):
        pack_signed(torch.zeros(2, 3, dtype=torch.int8), 3)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", [False, True])
@torch.inference_mode()
def test_gpu_quantized_linear(bits, dtype, activation):
    from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear

    torch.manual_seed(17)
    n, k = 41, 97
    group = 32
    q = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (n, k), dtype=torch.int8, device="cuda")
    scales = torch.rand(n, 4, device="cuda") * 0.03 + 0.001
    amax = torch.rand(k, device="cuda") * 2 + 0.5 if activation else None
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.01
    op = PackedLinear(
        q,
        scales,
        bits=bits,
        group_size=group,
        bias=bias,
        activation_amax=amax,
        round_weight_scale=activation,
    ).cuda()
    wscale = scales.to(dtype).float() if activation else scales
    weights = (q.float() * wscale.repeat_interleave(group, 1)[:, :k]).to(dtype)
    for rows in (1, 34, 97):
        x = torch.randn(rows, k, device="cuda", dtype=dtype) * 0.4
        reference_x = x
        if activation:
            step = (amax.to(dtype) / 127).clamp_min(1e-8)
            reference_x = (x / step).round().clamp(-127, 127) * step
        expected = torch.nn.functional.linear(reference_x, weights, bias)
        actual = op(x)
        tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 0.02}[dtype]
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert op.kernel_invocations == 3
    assert not any(t.is_floating_point() and t.shape == (n, k) for t in op.buffers())


@pytest.mark.gpu
@torch.inference_mode()
def test_packed_reload_empty_and_no_dense_cache():
    from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear

    codes = torch.randint(-7, 8, (17, 33), dtype=torch.int8)
    scales = torch.ones(17)
    first = PackedLinear(codes, scales, bits=4).cuda()
    second = PackedLinear(torch.zeros_like(codes), scales, bits=4).cuda()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 33, dtype=torch.float16, device="cuda")
    torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)
    assert first(x[:0]).shape == (0, 17)
    assert first.packed_weight.dtype == torch.uint8
    assert not list(first.parameters())
    assert not any(t.is_floating_point() and t.shape == codes.shape for t in first.buffers())
    with pytest.raises(ValueError, match="no CPU fallback"):
        first(x.cpu())
