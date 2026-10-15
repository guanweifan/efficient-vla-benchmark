import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.rotated_channel import (
    RotatedChannelPreparation,
    RotatedOutput,
)


def rotate(x, rotations):
    groups, width = rotations.shape[:2]
    shaped = x.reshape(-1, groups, width).transpose(0, 1)
    return torch.bmm(shaped, rotations).transpose(0, 1).reshape_as(x)


def test_invalid_parameters_and_cpu():
    for bits in (True, 3, 16):
        with pytest.raises(ValueError):
            RotatedChannelPreparation(bits)
    with pytest.raises(ValueError):
        RotatedChannelPreparation(8, qmin=-129)
    x, r, s = torch.zeros(1, 128), torch.ones(4, 32, 32), torch.ones(128)
    with pytest.raises(ValueError):
        RotatedChannelPreparation(8)(x, r, s)
    with pytest.raises(ValueError):
        RotatedOutput()(x, r)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "bits,amax,low_extra", [(2, False, 0), (4, True, 1), (8, False, 1), (8, True, 0)]
)
@torch.inference_mode()
def test_fused_and_native_short_paths_exact_dynamic_lengths(dtype, width, bits, amax, low_extra):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(8017)
    torch.backends.cuda.matmul.allow_tf32 = False
    k, high = 256, 2 ** (bits - 1) - 1
    low = -high - low_extra
    r = (torch.randn(k // width, width, width, device="cuda") / width**0.5).to(dtype)
    op = RotatedChannelPreparation(bits, qmin=low, amax=amax)
    restore = RotatedOutput()
    for m in (1, 15, 16, 17, 279, 280):
        x = torch.randn(1, m, k, device="cuda", dtype=dtype)
        # New contents must be read despite compiled dispatch reuse.
        scale = torch.rand(k, device="cuda") * 0.03 + 0.01
        step = (scale.to(dtype) / high).clamp_min(1e-8) if amax else scale.to(dtype)
        expected_codes = (rotate(x, r) / step).round().clamp(low, high).to(torch.int8)
        codes, prepared = op(x, r, scale)
        assert torch.equal(codes, expected_codes)
        assert torch.equal(prepared, (expected_codes.to(dtype) * step).half())
        half = x.half()
        for bias in (None, torch.randn(k, device="cuda", dtype=dtype) * 0.01):
            expected = rotate(half.to(dtype), r)
            if bias is not None:
                expected = expected + bias
            assert torch.equal(restore(half, r, bias), expected)
    assert len(op._compiled) == 1
    assert len(restore._compiled) == 2
    assert op(x[:, :0], r, scale)[1].shape == (1, 0, k)
    assert restore(half[:, :0], r).shape == (1, 0, k)
    with pytest.raises(ValueError):
        op(x, r, scale[:-1])
    with pytest.raises(ValueError):
        restore(half, r, scale[:-1])
    op.clear()
    restore.clear()
    assert not op._compiled and not restore._compiled
