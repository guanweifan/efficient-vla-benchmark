"""Optional indexed GEMMs: gather weight addresses inside the GPU kernel.

Weights stay dense in storage. Only selected rows/columns are loaded and multiplied;
this is structured execution, not dense GEMM followed by an output mask.
Token count is a non-specializing runtime argument: a new prompt length must not
trigger compilation after the common warmup. Feature widths and tiles stay fixed.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _linear(
    X,
    W,
    ROWS,
    COLS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    SELECT_ROWS: tl.constexpr,
    SELECT_COLS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    wn = tl.load(ROWS + n, n < N, other=0) if SELECT_ROWS else n
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + tl.arange(0, BK)
        wk = tl.load(COLS + k, k < K, other=0) if SELECT_COLS else k
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + wk[:, None] * W1 + wn[None, :] * W0, (k[:, None] < K) & (n[None, :] < N), 0)
        acc += tl.dot(x, w)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit(do_not_specialize=["M"])
def _gate_up(
    X,
    G,
    U,
    ROWS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    U0: tl.constexpr,
    U1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    wn = tl.load(ROWS + n, n < N, other=0)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        g = tl.load(G + k[:, None] * G1 + wn[None, :] * G0, (k[:, None] < K) & (n[None, :] < N), 0)
        u = tl.load(U + k[:, None] * U1 + wn[None, :] * U0, (k[:, None] < K) & (n[None, :] < N), 0)
        gate += tl.dot(x, g)
        up += tl.dot(x, u)
    # Preserve the native low-precision intermediate rounding convention.
    dtype: tl.constexpr = X.dtype.element_ty
    gate = gate.to(dtype).to(tl.float32)
    up = up.to(dtype).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(dtype).to(tl.float32)
    out = activated * up
    tl.store(Y + m[:, None] * N + n[None, :], out, (m[:, None] < M) & (n[None, :] < N))


def _validate(x, weight, indices):
    if (
        not x.is_cuda
        or weight.device != x.device
        or weight.ndim != 2
        or x.dtype not in (torch.float16, torch.bfloat16)
        or weight.dtype != x.dtype
        or x.ndim < 2
    ):
        raise ValueError("Indexed GEMM needs same-device CUDA FP16/BF16 inputs and weights")
    for index in indices:
        if index is not None and (
            index.ndim != 1
            or index.numel() == 0
            or index.dtype != torch.long
            or index.device != x.device
            or not index.is_contiguous()
        ):
            raise ValueError("Indices must be nonempty contiguous device-local int64 vectors")


def indexed_linear(x, weight, *, rows=None, columns=None):
    """The caller owns bounds/uniqueness of already-validated selection indices."""
    _validate(x, weight, (rows, columns))
    k = weight.shape[1] if columns is None else columns.numel()
    n = weight.shape[0] if rows is None else rows.numel()
    if x.shape[-1] != k:
        raise ValueError("Compact input width must match selected weight columns")
    flat = x.reshape(-1, k).contiguous()
    y = torch.empty((flat.shape[0], n), dtype=x.dtype, device=x.device)
    _linear[(triton.cdiv(flat.shape[0], 32), triton.cdiv(n, 64))](
        flat,
        weight,
        rows if rows is not None else flat,
        columns if columns is not None else flat,
        y,
        flat.shape[0],
        n,
        k,
        *weight.stride(),
        rows is not None,
        columns is not None,
        32,
        64,
        32,
        num_warps=4,
    )
    return y.reshape(*x.shape[:-1], n)


def indexed_gate_up(x, gate_weight, up_weight, rows):
    _validate(x, gate_weight, (rows,))
    _validate(x, up_weight, ())
    if rows is None or gate_weight.shape != up_weight.shape or x.shape[-1] != gate_weight.shape[1]:
        raise ValueError("Coupled gate/up weights must have equal shape and selected rows")
    n, k = rows.numel(), x.shape[-1]
    flat = x.reshape(-1, k).contiguous()
    y = torch.empty((flat.shape[0], n), dtype=x.dtype, device=x.device)
    _gate_up[(triton.cdiv(flat.shape[0], 32), triton.cdiv(n, 64))](
        flat,
        gate_weight,
        up_weight,
        rows,
        y,
        flat.shape[0],
        n,
        k,
        *gate_weight.stride(),
        *up_weight.stride(),
        32,
        64,
        32,
        num_warps=4,
    )
    return y.reshape(*x.shape[:-1], n)
