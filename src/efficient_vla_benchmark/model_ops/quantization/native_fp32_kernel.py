"""Decode packed weights for one native FP32 GEMM, never a persistent cache."""

import triton
import triton.language as tl


@triton.jit
def decode_fp32(
    W,
    S,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (n[:, None] < N) & (k[None, :] < K)
    packed = tl.load(W + n[:, None] + (k[None, :] // (8 // BITS)) * N, mask, other=0).to(tl.int32)
    code = (packed >> ((k[None, :] % (8 // BITS)) * BITS)) & ((1 << BITS) - 1)
    code = tl.where(code >= (1 << (BITS - 1)), code - (1 << BITS), code)
    scale = tl.load(S + n[:, None] * GROUPS + k[None, :] // GROUP, mask, other=0)
    tl.store(Y + n[:, None] * K + k[None, :], code.to(tl.float32) * scale, mask)
