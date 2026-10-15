import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.channel_linear import ChannelPackedLinear


def test_invalid_group_rejected():
    with pytest.raises(ValueError):
        ChannelPackedLinear(torch.zeros(128, 128, dtype=torch.int8), torch.ones(128), group_size=64)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [4, 8, 16])
@torch.inference_mode()
def test_unfused_reference_reset_scales_and_reload(dtype, bits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(573)
    codes = torch.randint(-8, 8, (256, 256), device="cuda", dtype=torch.int8)
    scales = torch.rand(256, 1, device="cuda") * 0.01 + 0.003
    step = torch.rand(256, device="cuda") * 0.03 + 0.01
    bias = torch.randn(256, device="cuda")
    op = ChannelPackedLinear(
        codes,
        scales,
        bias=bias,
        activation_step=step,
        activation_bits=bits,
        activation_qmin=-(2 ** (min(bits, 8) - 1)),
    )
    for m in (34, 279, 280):
        x = torch.randn(1, m, 256, device="cuda", dtype=dtype)
        if bits < 16:
            maximum = 2 ** (bits - 1) - 1
            q = (
                (x / op.activation_step.to(dtype))
                .round()
                .clamp(-maximum - 1, maximum)
                .to(torch.int8)
            )
            prepared = (q.to(dtype) * op.activation_step.to(dtype)).half()
        else:
            prepared = x.half()
        assert torch.equal(op(x), op.kernel(prepared).to(dtype))
        # Dynamic denoising tables must not be captured by dispatch caching.
        op.activation_step = op.activation_step * 1.01
    other = ChannelPackedLinear(
        codes.cpu(),
        scales.cpu(),
        bias=bias.cpu(),
        activation_step=step.cpu(),
        activation_bits=bits,
        activation_qmin=-(2 ** (min(bits, 8) - 1)),
    ).cuda()
    other.load_state_dict(op.state_dict())
    assert torch.equal(op(x), other(x))
    assert other(x[:, :0]).shape == (1, 0, 256)
    assert not list(op.parameters())
    assert op.execution_metadata()["accumulation"] == "float16"
