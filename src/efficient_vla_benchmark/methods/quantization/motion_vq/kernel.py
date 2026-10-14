"""Packed centroid gather inside GEMM; no full-weight reconstruction."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def centroid_gemm(
    X,
    INDICES,
    C,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    IBITS: tl.constexpr,
    BLOCK: tl.constexpr,
    CENTERS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    stride = N * IBITS // 8
    for index in range(tl.cdiv(K, BK)):
        k = index * BK + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0)
        byte = n[None, :] * IBITS // 8
        shift = n[None, :] * IBITS % 8
        offset = (k[:, None] // 2) * stride + byte
        first = tl.load(INDICES + offset, (k[:, None] < K) & (n[None, :] < N), other=0).to(tl.int32)
        if IBITS == 6:
            second = tl.load(
                INDICES + offset + 1,
                (k[:, None] < K) & (n[None, :] < N) & (shift + IBITS > 8),
                other=0,
            ).to(tl.int32)
            first = first | (second << 8)
        centroid = (first >> shift) & ((1 << IBITS) - 1)
        tile = (n[None, :] // BLOCK) * (K // BLOCK) + (k[:, None] // BLOCK)
        weight = tl.load(
            C + (tile * CENTERS + centroid) * 2 + (k[:, None] % 2),
            (k[:, None] < K) & (n[None, :] < N),
            other=0,
        ).to(Y.dtype.element_ty)
        acc += tl.dot(x, weight, allow_tf32=False)
    if HAS_BIAS:
        acc += tl.load(B + n, n < N, other=0).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))
