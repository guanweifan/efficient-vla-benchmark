"""Real sparse GEMM kernels against independent materialized-weight references."""

import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("column_major", [False, True])
def test_indexed_gemm_and_fused_gate_up(dtype_name, column_major):
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    from efficient_vla_benchmark.model_ops.indexed_linear import indexed_gate_up, indexed_linear

    torch.manual_seed(17)
    dtype = getattr(torch, dtype_name)
    x = torch.randn(1, 39, 96, device="cuda", dtype=dtype)
    w = torch.randn(137, 96, device="cuda", dtype=dtype) / 10
    if column_major:
        w = w.t().contiguous().t()
    rows = torch.randperm(137, device="cuda")[:75].sort().values
    columns = torch.randperm(96, device="cuda")[:51].sort().values
    for r, c in ((rows, None), (None, columns), (rows, columns), (None, None)):
        inp = x if c is None else x[..., c]
        weights = w if r is None else w[r]
        weights = weights if c is None else weights[:, c]
        actual = indexed_linear(inp, w, rows=r, columns=c)
        torch.testing.assert_close(actual, F.linear(inp, weights), atol=0.02, rtol=0.02)
    u = torch.randn_like(w) / 10
    actual = indexed_gate_up(x, w, u, rows)
    reference = F.silu(F.linear(x, w[rows])) * F.linear(x, u[rows])
    torch.testing.assert_close(actual, reference, atol=0.025, rtol=0.025)
