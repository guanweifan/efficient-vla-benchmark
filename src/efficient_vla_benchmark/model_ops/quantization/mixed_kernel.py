"""Single-bank W4 kernels; tokenwise A2/A4/A8 preparation and coalesced tile loads."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def prepare_tokens(X, Q, S, M, K: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, other=0).to(tl.float32)
    maximum: tl.constexpr = (1 << (BITS - 1)) - 1
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-5) / maximum
    normalized = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    q = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[normalized],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = tl.minimum(tl.maximum(q, -maximum), maximum) & ((1 << BITS) - 1)
    width: tl.constexpr = 8 // BITS
    grouped = tl.reshape(q, (BLOCK // width, width))
    packed = tl.sum(grouped << (tl.arange(0, width)[None, :] * BITS), 1).to(tl.uint8)
    j = tl.arange(0, BLOCK // width)
    tl.store(Q + row * (K // width) + j, packed, j < K // width)
    tl.store(S + row, scale)


@triton.jit
def expand_two(Q, P, SIZE, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Q + i // 2, i < SIZE, other=0).to(tl.int32)
    value = value >> ((i % 2) * 4)
    a, b = value & 3, (value >> 2) & 3
    a, b = tl.where(a >= 2, a + 12, a), tl.where(b >= 2, b + 12, b)
    tl.store(P + i, (a | (b << 4)).to(tl.uint8), i < SIZE)


@triton.jit
def scale_integer(SUMS, S, W, Y, SIZE, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(SUMS + i, i < SIZE, other=0).to(tl.float32)
    s = tl.load(S + i // N, i < SIZE, other=0)
    w = tl.load(W + i % N, i < SIZE, other=0)
    tl.store(Y + i, value * s * w, i < SIZE)


@triton.jit(do_not_specialize=["M"])
def packed_load(
    X,
    W,
    XS,
    WS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    if BITS == 8:
        acc = tl.zeros((BM, BN), tl.int32)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    scales = tl.load(WS + n, n < N, other=0)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + tl.arange(0, BK)
        half_k = block * (BK // 2) + tl.arange(0, BK // 2)
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0)
        # One K-contiguous load per physical byte, two signed nibbles in registers.
        packed = tl.load(
            W + n[:, None] * (K // 2) + half_k[None, :],
            (n[:, None] < N) & (half_k[None, :] < K // 2),
            other=0,
        ).to(tl.int32)
        pair = (packed[:, :, None] >> (tl.arange(0, 2)[None, None, :] * 4)) & 15
        q = tl.reshape(tl.where(pair >= 8, pair - 16, pair), (BN, BK))
        if BITS == 8:
            w = q.to(tl.int8)
        else:
            w = (q.to(tl.float32) * scales.to(tl.bfloat16).to(tl.float32)[:, None]).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc, allow_tf32=False)
    if BITS == 8:
        xs = tl.load(XS + m, m < M, other=0)
        output = acc.to(tl.float32) * xs[:, None] * scales[None, :]
    else:
        output = acc
    tl.store(Y + m[:, None] * N + n[None, :], output, (m[:, None] < M) & (n[None, :] < N))
