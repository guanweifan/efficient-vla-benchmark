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


@pytest.mark.parametrize("block_size,group_size", [(8, 8), (8, 16), (16, 8)])
def test_dynamic_group_solver_against_separate_block_reference(block_size, group_size):
    torch.manual_seed(107)
    w = torch.randn(7, 32)
    x = torch.randn(72, 32)
    h = 2 * x.T @ x / len(x)
    codes, scales = quantize(w, h, 4, block_size, group_size=group_size)
    h.diagonal().add_(h.diagonal().mean() * 0.01)
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    current = w.clone()
    reference, expected_scales = torch.empty_like(codes), torch.empty_like(scales)
    for start in range(0, 32, block_size):
        end = start + block_size
        working = current[:, start:end].clone()
        errors = []
        for column in range(start, end):
            if column % group_size == 0:
                group = current[:, column : column + group_size]
                lo = torch.minimum(group.min(1).values, torch.zeros(7))
                hi = torch.maximum(lo.abs(), group.max(1).values)
                lo = torch.where(lo < 0, -hi, lo)
                scale = (hi - lo) / 15
                expected_scales[:, column // group_size] = scale
            offset = column - start
            code = (working[:, offset] / scale).round().clamp(-8, 7)
            error = (working[:, offset] - code * scale) / upper[column, column]
            working[:, offset:] -= torch.outer(error, upper[column, column:end])
            errors.append(error)
            reference[:, column] = code.to(torch.int8)
        current[:, end:] -= torch.stack(errors, 1) @ upper[start:end, end:]
    torch.testing.assert_close(codes, reference, atol=0, rtol=0)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)


@pytest.mark.parametrize("group_size", [0, -1, True, 3, 2.5])
def test_group_geometry_rejected(group_size):
    with pytest.raises(ValueError, match="groups"):
        quantize(torch.randn(4, 16), torch.eye(16), 4, group_size=group_size)


@pytest.mark.gpu
def test_group_extrema_keep_author_device_division_rounding():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(370)
    weight = torch.randn(16, 128, device="cuda") * 0.05
    codes, scales = quantize(weight, torch.eye(128, device="cuda"), 4, group_size=128)
    maximum = weight.abs().amax(1)
    expected_scale = (2 * maximum) / torch.tensor(15, device="cuda")
    expected_codes = (weight / expected_scale[:, None]).round().clamp(-8, 7).to(torch.int8)
    torch.testing.assert_close(scales[:, 0], expected_scale, atol=0, rtol=0)
    assert torch.equal(codes, expected_codes)
