import numpy as np
import pytest
import torch

from efficient_vla_benchmark.methods.quantization.actquant.config import ActQuantConfig
from efficient_vla_benchmark.methods.quantization.actquant.implementations.openvla_oft import (
    calibrate,
)


def test_native_libero_gripper_transform_and_quantile_normalization():
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, -1] = [0, 1, 0.25, 0.75, -1, 2, 0, 1]
    stats = {"q01": [-1] * 6 + [0], "q99": [1] * 7, "mask": [True] * 6 + [False]}
    actual = calibrate.normalize_actions(actions, stats)
    np.testing.assert_array_equal(actual[:, -1], [1, 0, 0.75, 0.25, 1, 0, 1, 0])
    np.testing.assert_allclose(actual[:, :6], 0, atol=1e-7)
    np.testing.assert_array_equal(actions[:, -1], [0, 1, 0.25, 0.75, -1, 2, 0, 1])


@pytest.mark.parametrize(
    "value",
    [
        {"alpha": 0},
        {"alpha": 1},
        {"calibration_samples": 7},
        {"target_bpw": float("nan")},
        {"calibration_samples": True},
        {"extra": 1},
    ],
)
def test_config_rejects_disabled_core_or_invalid_budget(value):
    with pytest.raises(ValueError):
        ActQuantConfig.parse(value)


def test_complete_leaf_gradient_offload_matches_standard_fisher():
    torch.manual_seed(7)
    model = torch.nn.Linear(5, 3, bias=False)
    x, y = torch.randn(2, 5), torch.randn(2, 3)
    reference = torch.zeros_like(model.weight)
    for a, b in zip(x, y):
        (0.5 * (model(a) - b).abs().sum() + 0.5 * model(a).square().mean()).backward()
        reference.add_(model.weight.grad.square())
        model.weight.grad = None
    collected = torch.zeros_like(reference)
    calls = []

    def hook(par):
        collected.add_(par.grad.detach().float().square().cpu())
        calls.append(1)
        par.grad = None

    handle = model.weight.register_post_accumulate_grad_hook(hook)
    for a, b in zip(x, y):
        (0.5 * (model(a) - b).abs().sum() + 0.5 * model(a).square().mean()).backward()
    handle.remove()
    torch.testing.assert_close(collected, reference, rtol=0, atol=0)
    assert len(calls) == 2 and model.weight.grad is None
