import pytest
import torch

from efficient_vla_benchmark.methods.quantization.qvla.config import QVLAConfig
from efficient_vla_benchmark.methods.quantization.qvla.sensitivity import (
    channel_scores,
    hessian_scale,
)


@pytest.mark.parametrize(
    "parameters",
    [
        {"candidate_bits": [True, 16]},
        {"target_avg_bits": 16},
        {"calibration_samples": 5},
        {"percdamp": float("nan")},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        QVLAConfig.parse(parameters)


def test_batch_normalization_matches_official_recurrence():
    torch.manual_seed(9)
    batches = [torch.randn(1, n, 8) for n in (7, 19, 11)]
    h = torch.zeros(8, 8)
    count = 0
    total = torch.zeros_like(h)
    for x in batches:
        tmp = x.shape[0]
        h *= count / (count + tmp)
        count += tmp
        rows = x.reshape(-1, 8)
        scaled = (2 / count) ** 0.5 * rows.T
        h += scaled @ scaled.T
        total += rows.T @ rows
    diagonal = torch.arange(8)
    h[diagonal, diagonal] += 0.01 * h.diagonal().mean()
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    expected = upper.diagonal().square().clamp_min(1e-12).reciprocal()
    torch.testing.assert_close(hessian_scale(total, count, 0.01), expected)
    assert not torch.allclose(
        hessian_scale(total, sum(x.shape[1] for x in batches), 0.01), expected
    )


def test_dead_columns_and_row_quantizer():
    h = torch.diag(torch.tensor([2.0, 0.0, 4.0]))
    scale = hessian_scale(h, 2, 0.01)
    w = torch.tensor([[0.0, 1e-9, 2.0], [-1.0, 0.0, 1.0]])
    scores = channel_scores(w, scale, (0, 2, 4, 8, 16))
    assert scores[16] == [0.0, 0.0]
    torch.testing.assert_close(torch.tensor(scores[0]), (w.square() * scale).sum(1))
    assert all(x >= 0 for values in scores.values() for x in values)
