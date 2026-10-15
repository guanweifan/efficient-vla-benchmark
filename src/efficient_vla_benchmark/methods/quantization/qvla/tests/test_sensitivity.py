import pytest
import torch

from efficient_vla_benchmark.methods.quantization.qvla.config import QVLAConfig
from efficient_vla_benchmark.methods.quantization.qvla.sensitivity import (
    InputHessian,
    channel_scores,
    inverse_cholesky_diagonal,
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
    statistics = InputHessian(torch.nn.Linear(8, 3))
    for x in batches:
        tmp = x.shape[0]
        h *= count / (count + tmp)
        count += tmp
        rows = x.reshape(-1, 8)
        scaled = torch.sqrt(torch.tensor(2.0 / count)) * rows.T
        h += scaled @ scaled.T
        statistics.add_batch(x)
    assert statistics.examples == 3
    torch.testing.assert_close(statistics.matrix, h, atol=0, rtol=0)
    diagonal = torch.arange(8)
    h[diagonal, diagonal] += 0.01 * torch.diag(h).mean()
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    torch.testing.assert_close(
        inverse_cholesky_diagonal(statistics.matrix, 0.01), torch.diag(upper), atol=0, rtol=0
    )


def test_dead_columns_and_row_quantizer():
    h = torch.diag(torch.tensor([2.0, 0.0, 4.0]))
    diagonal = inverse_cholesky_diagonal(h, 0.01)
    w = torch.tensor([[0.0, 1e-9, 2.0], [-1.0, 0.0, 1.0]])
    scores = channel_scores(w, diagonal, (0, 2, 4, 8, 16))
    assert scores[16] == [0.0, 0.0]
    torch.testing.assert_close(torch.tensor(scores[0]), (w.square() / diagonal.square()).sum(1))
    assert all(x >= 0 for values in scores.values() for x in values)


@pytest.mark.gpu
@pytest.mark.parametrize("convolution", [False, True])
def test_literal_cuda_statistics_and_proxy_match_reference(convolution):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(81)
    layer = (
        torch.nn.Conv2d(4, 9, 3, padding=1, device="cuda", dtype=torch.bfloat16)
        if convolution
        else torch.nn.Linear(1024, 9, device="cuda", dtype=torch.bfloat16)
    )
    stats = InputHessian(layer)
    h = torch.zeros(stats.columns, stats.columns, device="cuda")
    examples = 0
    for batch in (1, 2, 1):
        shape = (batch, 4, 11, 13) if convolution else (batch, 17, 1024)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        if convolution:
            rows = torch.nn.Unfold(3, padding=1)(x).permute(1, 0, 2).flatten(1)
        else:
            rows = x.reshape(-1, 1024).T
        h *= examples / (examples + batch)
        examples += batch
        scaled = torch.sqrt(torch.tensor(2.0 / examples, device="cuda")) * rows.float()
        h += scaled @ scaled.T
        stats.add_batch(x)
    torch.testing.assert_close(stats.matrix.cuda(), h, atol=0, rtol=0)
    diagonal = torch.arange(stats.columns, device="cuda")
    h[torch.diag(h) == 0, torch.diag(h) == 0] = 1
    h[diagonal, diagonal] += 0.01 * torch.diag(h).mean()
    reference_diag = torch.diag(
        torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    )
    actual_diag = inverse_cholesky_diagonal(stats.matrix.cuda(), 0.01)
    torch.testing.assert_close(actual_diag, reference_diag, atol=0, rtol=0)
    weight = layer.weight.detach().flatten(1).float()
    scores = channel_scores(layer.weight, actual_diag, (0, 2, 4, 8, 16))
    for bit in (0, 2, 4, 8):
        quantized = []
        for row in weight:
            if bit == 0:
                quantized.append(torch.zeros_like(row))
            else:
                qmax = 2 ** (bit - 1) - 1
                scale = row.abs().max().clamp_min(1e-6) / qmax
                quantized.append((row / scale).round().clamp(-qmax - 1, qmax) * scale)
        expected = (
            (weight - torch.stack(quantized)).square() / reference_diag.square().clamp_min(1e-12)
        ).sum(1)
        torch.testing.assert_close(torch.tensor(scores[bit]), expected.cpu(), atol=0, rtol=0)


def test_schema_four_artifact_is_not_silently_reinterpreted(tmp_path):
    import json

    from efficient_vla_benchmark.methods.quantization.qvla.artifact import read_manifest

    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(schema_version=4, algorithm="qvla", complete=True))
    )
    with pytest.raises(ValueError, match="schema 5"):
        read_manifest(tmp_path, QVLAConfig())
