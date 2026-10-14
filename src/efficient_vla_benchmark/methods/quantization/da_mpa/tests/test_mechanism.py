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
