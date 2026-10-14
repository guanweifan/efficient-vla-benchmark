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
def test_unsigned_index_packing_roundtrip(bits):
    torch.manual_seed(9)
    x = torch.randint(1 << bits, (128, 256))
    packed = pack_indices(x, bits)
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
@torch.inference_mode()
def test_centroid_kernel_matches_reconstructed_oracle(block, centers, bits):
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
    x = torch.randn(35, k, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(op(x), F.linear(x, weight), atol=0.01, rtol=0.01)
    assert op.kernel_invocations == 1
