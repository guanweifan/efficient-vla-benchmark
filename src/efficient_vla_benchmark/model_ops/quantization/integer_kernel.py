"""INT8 Tensor Core GEMM with token/channel scales outside the K reduction.

These equations are NOT compatible with an arbitrary per-input-channel activation
quantizer. Weight-only and channel-activation callers must not silently use them.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def quantize_tokens(X, Q, S, M, K: tl.constexpr, FLOOR: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), FLOOR) / 127.0
    normalized = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # round-to-nearest-even matches torch.round, including exact half-way cases.
    rounded = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[normalized],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = tl.minimum(tl.maximum(rounded, -127), 127).to(tl.int8)
    tl.store(Q + row * K + k, q, k < K)
    tl.store(S + row, scale)


@triton.jit(do_not_specialize=["M"])
def integer_gemm(
    X,
    W,
    XS,
    WS,
    BIAS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.int32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0)
        packed = tl.load(
            W + (k[:, None] // (8 // BITS)) * N + n[None, :],
            (k[:, None] < K) & (n[None, :] < N),
            other=0,
        )
        if BITS == 8:
            w = packed.to(tl.int8)
        else:
            q = (packed.to(tl.int32) >> ((k[:, None] % 2) * 4)) & 15
            w = tl.where(q >= 8, q - 16, q).to(tl.int8)
        # No floating inputs to this dot: signed INT8 multiplication, INT32 sum.
        acc = tl.dot(x, w, acc)
    xs = tl.load(XS + m, m < M, other=0)
    ws = tl.load(WS + n, n < N, other=0)
    result = acc.to(tl.float32) * xs[:, None] * ws[None, :]
    if HAS_BIAS:
        result += tl.load(BIAS + n, n < N, other=0).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], result, (m[:, None] < M) & (n[None, :] < N))
