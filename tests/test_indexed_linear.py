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


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_token_length_is_runtime_only_and_matches_static_specialization(dtype_name):
    """New prompt lengths must not compile in a warmed measurement window."""
    import types

    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    import triton.language as tl

    from efficient_vla_benchmark.model_ops.indexed_linear import (
        _gate_up,
        _linear,
        indexed_gate_up,
        indexed_linear,
    )

    def static_reference(kernel):
        fn = kernel.fn
        copy = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__)
        copy.__annotations__ = {**fn.__annotations__, "M": tl.constexpr}
        return triton.jit(copy)

    static_linear, static_gate = static_reference(_linear), static_reference(_gate_up)
    assert 5 in static_linear.constexprs and 5 not in _linear.constexprs
    assert 5 in static_gate.constexprs and 5 not in _gate_up.constexprs
    torch.manual_seed(23)
    dtype = getattr(torch, dtype_name)
    weight = torch.randn(137, 96, device="cuda", dtype=dtype).t().contiguous().t()
    up = torch.randn_like(weight)
    rows = torch.randperm(137, device="cuda")[:75].sort().values
    device = torch.cuda.current_device()
    warmed_counts = None
    for length in (39, 47, 64, 1):
        x = torch.randn(1, length, 96, device="cuda", dtype=dtype)
        expected = torch.empty((length, 75), device="cuda", dtype=dtype)
        grid = (triton.cdiv(length, 32), triton.cdiv(75, 64))
        static_linear[grid](
            x,
            weight,
            rows,
            x,
            expected,
            length,
            75,
            96,
            *weight.stride(),
            True,
            False,
            32,
            64,
            32,
            num_warps=4,
        )
        actual = indexed_linear(x, weight, rows=rows)
        assert torch.equal(actual.reshape_as(expected), expected)
        static_gate[grid](
            x,
            weight,
            up,
            rows,
            expected,
            length,
            75,
            96,
            *weight.stride(),
            *up.stride(),
            32,
            64,
            32,
            num_warps=4,
        )
        actual = indexed_gate_up(x, weight, up, rows)
        assert torch.equal(actual.reshape_as(expected), expected)
        counts = (len(_linear.cache[device]), len(_gate_up.cache[device]))
        if warmed_counts is not None:
            assert counts == warmed_counts
        warmed_counts = counts
