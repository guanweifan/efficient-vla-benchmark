import numpy as np
import pytest
import torch

from efficient_vla_benchmark.methods.quantization.da_mpa.config import DAMPAConfig
from efficient_vla_benchmark.methods.quantization.da_mpa.drift import (
    drift_weights,
    normalize_actions,
    retained_blocks,
    structural_jacobian,
)


def test_virtual_chain_jacobian_matches_finite_difference():
    q = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)

    def end(x):
        theta = x.cumsum(0)
        return torch.stack((theta.cos().sum(), theta.sin().sum(), x.sum()))

    j, _ = structural_jacobian(q[None], 0.0003)
    cols = []
    for i in range(7):
        step = torch.zeros_like(q)
        step[i] = 1e-6
        cols.append((end(q + step) - end(q - step)) / 2e-6)
    torch.testing.assert_close(j[0], torch.stack(cols, 1), atol=1e-8, rtol=1e-8)
    weights = drift_weights(q[None], DAMPAConfig())
    assert (weights > 0).all()
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_native_action_normalization_and_full_block_retention():
    stats = dict(q01=[-1] * 7, q99=[1] * 7, mask=[True] * 6 + [False])
    raw = np.zeros((16, 7), dtype=np.float32)
    raw[:8, 6] = -1
    raw[8:, 6] = 1
    normalized = normalize_actions(raw, stats)
    np.testing.assert_array_equal(normalized[:8, 6], np.ones(8))
    np.testing.assert_array_equal(normalized[8:, 6], np.zeros(8))
    keep = retained_blocks({i: float(i) for i in range(12)}, range(12), 0.3, 2)
    assert keep == {7, 8, 9, 10, 11}
    assert (
        DAMPAConfig.parse({"retention_ratio": 0.15}).calibration_parameters()
        == DAMPAConfig().calibration_parameters()
    )
    with pytest.raises(ValueError):
        DAMPAConfig.parse({"retention_ratio": 1})


def test_epsilon_probe_restores_grad_flags_without_weight_updates():
    from types import SimpleNamespace

    from efficient_vla_benchmark.methods.quantization.da_mpa.implementations.cogact import (
        calibrate as implementation,
    )

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(7, 7)
            self.proj.requires_grad_(False)

        def forward(self, x, t, z):
            return self.proj(x) + z.mean() * 0

    class D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Net()
            self.diffusion = SimpleNamespace(num_timesteps=10, q_sample=lambda x, t, n: x + n)

    d = D()
    before = d.net.proj.weight.detach().clone()
    scores, steps = implementation.epsilon_sensitivity(
        d,
        [torch.ones(1, 8)] * 4,
        [np.zeros((16, 7))] * 4,
        [("p", d.net.proj)],
        torch.ones(7),
        [0, 1, 2, 3],
        0,
    )
    assert scores["p"] > 0 and len(steps) == 4
    assert not d.net.proj.weight.requires_grad and d.net.proj.weight.grad is None
    torch.testing.assert_close(d.net.proj.weight, before, atol=0, rtol=0)


def test_runtime_is_explicit_but_does_not_change_native_calibration_identity():
    native = DAMPAConfig()
    packed = DAMPAConfig.parse({"runtime": "tinychat_fp16"})
    assert packed.calibration_parameters() == native.calibration_parameters()
    assert "runtime" not in packed.calibration_parameters()
    assert native.runtime == "tile_float"
    with pytest.raises(ValueError, match="runtime"):
        DAMPAConfig.parse({"runtime": "auto"})


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 17, 280])
def test_fp16_packed_matches_explicit_channel_quantizer_and_native_bias(dtype, rows):
    from efficient_vla_benchmark.methods.quantization.da_mpa.packed import DAMPALinear

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator(device="cuda").manual_seed(17)
    codes = torch.randint(-8, 8, (128, 256), generator=generator, device="cuda", dtype=torch.int8)
    scales = torch.linspace(0.005, 0.02, 128, device="cuda")
    amax = torch.linspace(0.5, 3.0, 256, device="cuda")
    bias = torch.linspace(-0.03, 0.03, 128, device="cuda", dtype=dtype)
    module = DAMPALinear(codes, scales, amax, bias)
    x = torch.randn(rows, 256, generator=generator, device="cuda", dtype=dtype)
    step = (amax.to(dtype) / 127).clamp_min(1e-8)
    quantized = (x / step).round().clamp(-127, 127).to(torch.int8)
    expected = module.linear.kernel((quantized.to(dtype) * step).half()).to(dtype) + bias
    actual = module(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == dtype and module.linear.kernel_invocations == 1
    assert torch.equal(module.bias, bias)
    reloaded = DAMPALinear(codes, scales, amax, bias)
    reloaded.load_state_dict(module.state_dict())
    torch.testing.assert_close(reloaded(x), actual, atol=0, rtol=0)
    assert module(x[:0]).shape == (0, 128)
