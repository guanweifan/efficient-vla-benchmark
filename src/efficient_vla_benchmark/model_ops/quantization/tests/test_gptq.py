import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.gptq import quantize, symmetric_scales


@pytest.mark.parametrize("bits", [4, 8])
@torch.inference_mode()
def test_global_gptq_matches_unblocked_column_oracle(bits):
    torch.manual_seed(17)
    w = torch.randn(13, 32)
    x = torch.randn(80, 32)
    h = 2 * x.T @ x / len(x)
    actual, scale = quantize(w, h, bits, 8, 0.01)
    reference = w.clone()
    hh = h.clone()
    hh.diagonal().add_(0.01 * hh.diagonal().mean())
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hh)), upper=True)
    codes = []
    for col in range(32):
        code = (reference[:, col] / scale).round().clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        err = (reference[:, col] - code * scale) / upper[col, col]
        reference[:, col:] -= err[:, None] * upper[col, col:][None, :]
        codes.append(code.to(torch.int8))
    torch.testing.assert_close(actual, torch.stack(codes, 1), atol=0, rtol=0)


def test_dead_columns_and_invalid_input():
    w = torch.randn(8, 16)
    h = torch.eye(16)
    h[2, 2] = 0
    codes, scales = quantize(w, h, 4, 8)
    assert not codes[:, 2].any()
    assert (scales > 0).all()
    with pytest.raises(ValueError):
        quantize(w, torch.eye(7), 4)


def test_standard_gptq_symmetric_endpoint():
    w = torch.tensor([[-1.0, 1.0], [0.0, 0.0], [1.0, 2.0]])
    torch.testing.assert_close(symmetric_scales(w, 4), torch.tensor([2 / 15, 2 / 15, 2 / 15]))
