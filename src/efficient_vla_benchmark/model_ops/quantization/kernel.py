"""Packed-weight GEMM with tile-local dequantization, not integer-MMA W4A8.

Per-input-channel activation scales remain inside the K reduction. In particular,
they are never replaced with a group maximum to fit an INT8 epilogue.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def packed_gemm(
    X,
    W,
    WS,
    AS,
    BIAS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    QUANT_A: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANSPOSED_STORAGE: tl.constexpr = False,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    dtype = Y.dtype.element_ty
    stride = tl.cdiv(K, 8 // BITS)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0)
        if QUANT_A:
            a_scale = tl.load(AS + k, k < K, other=0).to(dtype)
            x = (x.to(dtype).to(tl.float32) * a_scale[None, :].to(tl.float32)).to(dtype)
        if TRANSPOSED_STORAGE:
            weight_ptr = W + n[None, :] + (k[:, None] // (8 // BITS)) * N
        else:
            weight_ptr = W + n[None, :] * stride + k[:, None] // (8 // BITS)
        packed = tl.load(
            weight_ptr,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        ).to(tl.int32)
        q = (packed >> ((k[:, None] % (8 // BITS)) * BITS)) & ((1 << BITS) - 1)
        q = tl.where(q >= (1 << (BITS - 1)), q - (1 << BITS), q)
        scale = tl.load(
            WS + n[None, :] * GROUPS + k[:, None] // GROUP,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        if ROUND_SCALE:
            scale = scale.to(dtype).to(tl.float32)
        w = (q.to(tl.float32) * scale).to(dtype)
        acc += tl.dot(x, w, allow_tf32=False)
    if HAS_BIAS:
        acc += tl.load(BIAS + n, n < N, other=0).to(dtype).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))
