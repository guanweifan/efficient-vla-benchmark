import pytest
import torch

from efficient_vla_benchmark.model_ops.block_rotation import BlockRotation


def test_cpu_rejected():
    with pytest.raises(ValueError):
        BlockRotation()(torch.zeros(1, 128), torch.eye(16).repeat(8, 1, 1))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [16, 32, 64, 128])
@pytest.mark.parametrize("permuted", [False, True])
@torch.inference_mode()
def test_block_products_dynamic_lengths_and_cache(dtype, width, permuted):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(172)
    k = 512
    blocks = (
        torch.linalg.qr(torch.randn(k // width, width, width, device="cuda"))
        .Q.to(dtype)
        .contiguous()
    )
    permutation = torch.randperm(k, device="cuda") if permuted else None
    op = BlockRotation()
    for m in (1, 15, 16, 17, 34, 279, 280, 593, 597):
        x = torch.randn(1, m, k, device="cuda", dtype=dtype)
        selected = x.index_select(-1, permutation) if permuted else x
        expected = (
            torch.bmm(selected.reshape(-1, k // width, width).transpose(0, 1), blocks)
            .transpose(0, 1)
            .reshape_as(x)
        )
        actual = op(x, blocks, permutation)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(op._compiled) == int(m > 16)
    assert op(x[:, :0], blocks, permutation).shape == (1, 0, k)
    with pytest.raises(ValueError):
        op(x, blocks.double(), permutation)
    op.clear()
    assert not op._compiled
