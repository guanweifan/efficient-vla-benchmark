import pytest
import torch
from torch import nn
from torch.nn import functional as F

from efficient_vla_benchmark.methods.quantization.eaqvla.config import EaqVLAConfig
from efficient_vla_benchmark.methods.quantization.eaqvla.target import flatten_input, targets


def test_config_and_modality_boundaries():
    with pytest.raises(ValueError):
        EaqVLAConfig.parse({"vision_bits": 3})
    with pytest.raises(ValueError):
        EaqVLAConfig.parse({"calibration_samples": 5})
    model = nn.Module()
    model.vision = nn.Linear(8, 8)
    model.language = nn.Sequential(nn.Linear(8, 8))
    model.projector = nn.Linear(8, 8)
    model.action = nn.Linear(8, 7)
    found = targets(model, "vision", "language")
    assert set(found) == {"vision", "language.0"}
    assert found["vision"][1] == "vision"


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_patch_conv_matches_independent_quantized_weight(dtype):
    from efficient_vla_benchmark.methods.quantization.eaqvla.packed import PackedPatchConv

    torch.manual_seed(41)
    native = nn.Conv2d(3, 16, 4, stride=4).cuda().to(dtype)
    codes = torch.randint(-8, 8, (16, 48), dtype=torch.int8)
    scales = torch.rand(16) * 0.01 + 0.001
    entry = dict(codes=codes, scales=scales, bias=native.bias.detach().cpu())
    op = PackedPatchConv(native, entry, 4).cuda()
    x = torch.randn(2, 3, 16, 16, device="cuda", dtype=dtype)
    reference = F.conv2d(
        x, (codes.float() * scales[:, None]).reshape(16, 3, 4, 4).to(x), native.bias, stride=4
    )
    tol = 0.01 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(op(x), reference, atol=tol, rtol=tol)
    assert flatten_input(native, x).shape == (32, 48)
