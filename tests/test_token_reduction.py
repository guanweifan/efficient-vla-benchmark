"""Merge-result boundary checks independent of any paper."""

import pytest
import torch

from efficient_vla_benchmark.model_ops.token_reduction import TokenReduction


def test_causal_merge_preserves_nonvisual_and_original_anchor_positions():
    module = pytest.importorskip("efficient_vla_backend_cogact.decoder")
    adapter = object.__new__(module.CausalTokenAdapter)
    adapter.state = module.DecoderState(module.TokenLayout(8, patches_per_view=4), torch.arange(8))
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    keep = torch.tensor([True, False, True, False, True, True, True, True])
    merged = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    actual = adapter._select(hidden, TokenReduction(keep, merged))
    assert adapter.state.positions.tolist() == [0, 2, 4, 5, 6, 7]
    torch.testing.assert_close(actual[0, 1:3], merged)
    torch.testing.assert_close(actual[0, [0, 3, 4, 5]], hidden[0, [0, 5, 6, 7]])
    torch.testing.assert_close(hidden, torch.arange(24.0).reshape(1, 8, 3))


def test_causal_merge_rejects_wrong_dtype_and_nonvisual_deletion():
    module = pytest.importorskip("efficient_vla_backend_cogact.decoder")
    for bad in ("dtype", "protected"):
        adapter = object.__new__(module.CausalTokenAdapter)
        adapter.state = module.DecoderState(
            module.TokenLayout(8, patches_per_view=4), torch.arange(8)
        )
        hidden = torch.zeros(1, 8, 3)
        keep = torch.ones(8, dtype=torch.bool)
        if bad == "protected":
            keep[-1] = False
        with pytest.raises(ValueError):
            adapter._select(hidden, TokenReduction(keep, torch.zeros(4, 3, dtype=torch.float64)))
