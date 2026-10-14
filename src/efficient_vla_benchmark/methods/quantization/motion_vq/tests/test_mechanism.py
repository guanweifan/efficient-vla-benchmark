import numpy as np
import pytest
import torch
from torch.nn import functional as F

from efficient_vla_benchmark.methods.quantization.motion_vq.codebook import (
    pack_indices,
    unpack_indices,
    weighted_kmeans,
)
from efficient_vla_benchmark.methods.quantization.motion_vq.config import MotionVQConfig
from efficient_vla_benchmark.methods.quantization.motion_vq.router import MotionRouter


@pytest.mark.parametrize("bits", [6, 8])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8])
def test_unsigned_index_packing_roundtrip(bits, dtype):
    torch.manual_seed(9)
    x = torch.randint(1 << bits, (128, 256))
    x[0, :2] = torch.tensor([0, (1 << bits) - 1])
    packed = pack_indices(x.to(dtype), bits)
    assert packed.numel() == x.numel() * bits // 8
    torch.testing.assert_close(unpack_indices(packed, bits), x, atol=0, rtol=0)


def test_router_uses_executed_prefix_end_and_reset():
    r = MotionRouter(0.8)
    assert not r.use_low
    r.observe(np.array([1.0, 0, 0, 0, 0, 0, 0]), 1)
    assert not r.use_low
    r.observe(np.array([0.8, 0, 0, 0, 0, 0, 0]), 0)
    assert r.use_low
    r.observe(np.zeros(7), 0)
    assert not r.use_low
    r.reset()
    assert r.last_magnitude is None
    with pytest.raises(ValueError):
        MotionVQConfig.parse({"threshold": float("nan")})


def test_weighted_kmeans_keeps_public_update_order():
    torch.manual_seed(2)
    vectors = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [6.0, 7.0], [8.0, 9.0]]])
    weights = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [4.0, 1.0], [2.0, 3.0]]])
    books, indices = weighted_kmeans(vectors, weights, 1, 1)
    torch.testing.assert_close(books[0, 0], (vectors * weights).sum(1)[0] / weights.sum(1)[0])
    assert not indices.any()


@pytest.mark.gpu
@pytest.mark.parametrize("block,centers,bits", [(128, 64, 6), (256, 256, 8)])
@pytest.mark.parametrize("m", [35, 280, 600, 985])
@torch.inference_mode()
def test_centroid_kernel_matches_reconstructed_oracle(block, centers, bits, m):
    import triton

    from efficient_vla_benchmark.methods.quantization.motion_vq.kernel import centroid_gemm
    from efficient_vla_benchmark.methods.quantization.motion_vq.packed import PackedCodebookLinear

    torch.manual_seed(13)
    n, k = block * 2, block * 2
    books = (torch.randn(4, centers, 2) * 0.02).bfloat16()
    indices = torch.randint(centers, (k // 2, n))
    entry = dict(
        codebooks=books,
        indices=pack_indices(indices, bits),
        index_bits=bits,
        centers=centers,
        block_size=block,
        shape=(n, k),
        native_dtype="torch.bfloat16",
    )
    op = PackedCodebookLinear(entry).cuda()
    rows = torch.arange(n)[:, None]
    cols = torch.arange(k)[None, :]
    tile = (rows // block) * (k // block) + cols // block
    weight = books[tile, indices.T[:, cols[0] // 2], cols % 2].cuda()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    actual = op(x)
    torch.testing.assert_close(actual, F.linear(x, weight), atol=0.01, rtol=0.01)
    previous = torch.empty_like(actual)
    centroid_gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        x,
        op.indices,
        op.codebooks,
        op.codebooks,
        previous,
        m,
        n,
        k,
        bits,
        block,
        centers,
        False,
        32,
        64,
        32,
        num_warps=4,
    )
    torch.testing.assert_close(actual, previous, atol=0, rtol=0)
    assert op.kernel_invocations == 1


@pytest.mark.gpu
@pytest.mark.parametrize("block,centers,bits", [(128, 64, 6), (256, 256, 8)])
@torch.inference_mode()
def test_calibration_uint8_storage_reaches_packed_kernel(block, centers, bits):
    from efficient_vla_benchmark.methods.quantization.motion_vq.codebook import quantize_weight
    from efficient_vla_benchmark.methods.quantization.motion_vq.packed import PackedCodebookLinear

    torch.manual_seed(41)
    weight = (torch.randn(block, block, device="cuda") * 0.02).bfloat16()
    config = MotionVQConfig(kmeans_iterations=2, block_batch_size=1)
    entry = quantize_weight(weight, torch.ones(block, device="cuda"), block, centers, config)
    assert entry["indices"].dtype == torch.uint8
    indices = unpack_indices(entry["indices"], bits).T.repeat_interleave(2, dim=1)
    cols = torch.arange(block)[None, :] % 2
    dense = entry["codebooks"][0, indices, cols].cuda()
    op = PackedCodebookLinear(entry).cuda()
    x = torch.randn(35, block, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(op(x), F.linear(x, dense), atol=0.01, rtol=0.01)
