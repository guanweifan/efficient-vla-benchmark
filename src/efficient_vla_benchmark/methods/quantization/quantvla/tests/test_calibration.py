import pytest
import torch

from efficient_vla_benchmark.methods.quantization.quantvla.config import QuantVLAConfig
from efficient_vla_benchmark.methods.quantization.quantvla.packed import BlockRotation
from efficient_vla_benchmark.methods.quantization.quantvla.transforms import (
    attention_scales,
    build_transform,
    quantize_mse,
    transformed_weight,
)


@pytest.mark.parametrize(
    "parameters",
    [
        {"runtime": "fake"},
        {"block_size": True},
        {"calibration_samples": 3},
        {"alpha_min": 2.0},
        {"energy_smoothing": float("nan")},
    ],
)
def test_config_rejects_invalid(parameters):
    with pytest.raises(ValueError):
        QuantVLAConfig.parse(parameters)


def test_energy_smoothing_does_not_rescale_activation():
    torch.manual_seed(10)
    w = torch.randn(64, 64)
    t = build_transform(w, QuantVLAConfig(block_size=16, enable_permutation=True))
    assert torch.equal(t.smoothing, torch.ones(64))
    x = torch.randn(3, 64)
    xr = BlockRotation(t.input_rotations)(x.index_select(1, t.permutation))
    y = BlockRotation(t.output_rotations)(xr @ transformed_weight(w, t).T)
    torch.testing.assert_close(y, x @ w.T, atol=2e-5, rtol=2e-5)


def test_runtime_change_requires_distinct_calibration_identity():
    native = QuantVLAConfig()
    packed = QuantVLAConfig.parse({"runtime": "tinychat_fp16"})
    assert "runtime" not in native.calibration_parameters()
    assert packed.calibration_parameters()["runtime"] == "tinychat_fp16"
    assert native.execution() != packed.execution()


def test_mse_scale_matches_author_grid():
    torch.manual_seed(11)
    w = torch.randn(13, 41)
    q, scale = quantize_mse(w)
    candidates = (w.abs().amax(1) / 7).clamp_min(1e-8)[:, None] / torch.tensor(
        [0.5, 0.75, 1.0, 1.25, 1.5]
    )
    quant = (w[:, None, :] / candidates[:, :, None]).round().clamp(-8, 7)
    errors = (quant * candidates[:, :, None] - w[:, None, :]).square().mean(2)
    idx = errors.argmin(1)
    assert torch.equal(scale, candidates[torch.arange(13), idx])
    assert torch.equal(q, quant[torch.arange(13), idx].to(torch.int8))


def test_atm_and_ohb_have_distinct_author_neutral_rules():
    t = {"logits": torch.tensor([2.0, 1.01, 0.1]), "output": torch.tensor(2.0)}
    s = {"logits": torch.ones(3), "output": torch.tensor(1.0)}
    out = attention_scales(t, s, QuantVLAConfig())
    torch.testing.assert_close(torch.tensor(out["alpha"]), torch.tensor([1.4, 1.0, 0.7]))
    assert out["beta"] == pytest.approx(torch.tensor(0.3).exp().item())


@pytest.mark.gpu
def test_frozen_step_supports_negative_128():
    from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear

    q = torch.ones((16, 32), dtype=torch.int8, device="cuda")
    op = PackedLinear(
        q,
        torch.ones(16, device="cuda"),
        bits=4,
        activation_step=torch.ones(32, device="cuda"),
        activation_qmin=-128,
    ).cuda()
    x = torch.full((2, 32), -200.0, device="cuda")
    assert torch.equal(op(x), torch.full((2, 16), -4096.0, device="cuda"))
