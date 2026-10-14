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


@pytest.mark.gpu
@pytest.mark.parametrize("activation_bits", [2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_activation_precisions_use_declared_codes_and_native_bypass(activation_bits, dtype):
    torch.manual_seed(117)
    codes = torch.randint(-8, 8, (64, 128), dtype=torch.int8, device="cuda")
    weights = torch.rand(64, device="cuda") * 0.01 + 0.001
    step = torch.rand(128, device="cuda") * 0.02 + 0.01
    qmax = (1 << (min(activation_bits, 8) - 1)) - 1
    op = PackedLinear(
        codes,
        weights,
        bits=4,
        activation_bits=activation_bits,
        activation_step=step,
        activation_qmin=-qmax - 1,
    ).cuda()
    x = torch.randn(33, 128, device="cuda", dtype=dtype)
    ref_x = (
        x
        if activation_bits == 16
        else ((x / step.to(dtype)).round().clamp(-qmax - 1, qmax) * step.to(dtype))
    )
    ref = torch.nn.functional.linear(ref_x, (codes.float() * weights[:, None]).to(dtype))
    tol = 0.01 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(op(x), ref, atol=tol, rtol=tol)
    assert op.execution_metadata()["activation_bits"] == activation_bits


@pytest.mark.parametrize(
    "arguments",
    [
        dict(activation_bits=3),
        dict(activation_bits=True),
        dict(activation_bits=4, activation_qmin=-128),
    ],
)
def test_invalid_activation_range_rejected(arguments):
    with pytest.raises(ValueError):
        PackedLinear(torch.zeros((16, 32), dtype=torch.int8), torch.ones(16), bits=4, **arguments)
