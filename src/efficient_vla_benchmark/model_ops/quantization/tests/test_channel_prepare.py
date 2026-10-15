import numpy as np
import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.channel_prepare import ChannelPreparation


def test_invalid_parameters_and_cpu_rejected():
    for bits in (True, 3, 16):
        with pytest.raises(ValueError):
            ChannelPreparation(bits)
    with pytest.raises(ValueError):
        ChannelPreparation(4, qmin=-9)
    with pytest.raises(ValueError):
        ChannelPreparation(4)(torch.zeros(1, 128), torch.ones(128))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("amax", [False, True])
@pytest.mark.parametrize("full_negative", [False, True])
@torch.inference_mode()
def test_native_rounding_dynamic_lengths_and_cache(dtype, bits, amax, full_negative):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(20261015)
    high = 2 ** (bits - 1) - 1
    low = -high - int(full_negative)
    op = ChannelPreparation(bits, qmin=low, amax=amax)
    for m in (1, 34, 279, 280, 593, 597):
        scale = torch.rand(128, device="cuda") * 0.04 + 0.005
        x = (torch.randn(m, 128, device="cuda") * 0.07).to(dtype)
        step = (scale.to(dtype) / high).clamp_min(1e-8) if amax else scale.to(dtype)
        boundary = ((torch.arange(128, device="cuda") % 17 - 8).float() + 0.5).to(dtype) * step
        x[0] = boundary
        for row, direction in [(1, float("inf")), (2, -float("inf"))]:
            if m <= row:
                continue
            if dtype == torch.float16:
                x[row] = torch.from_numpy(
                    np.nextafter(boundary.cpu().numpy(), np.float16(direction))
                ).cuda()
            else:
                x[row] = torch.nextafter(boundary, torch.full_like(boundary, direction))
        expected = (x / step).round().clamp(low, high).to(torch.int8)
        codes, prepared = op(x, scale)
        assert torch.equal(codes, expected)
        assert torch.equal(prepared, (expected.to(dtype) * step).half())
        assert len(op._compiled) == 1
    assert op(x[:0], scale)[1].shape == (0, 128)
    with pytest.raises(ValueError):
        op(x, scale[:-1])
    op.clear()
    assert not op._compiled
