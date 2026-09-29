"""Independent NumPy oracles and explicit undefined-case boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import ILSAConfig
from ..selection import select_tokens, semantic_scores, text_anchors, view_diversity
from ..strategy import ILSAStrategy


def softmax(x):
    value = np.exp(x - x.max(axis=-1, keepdims=True))
    return value / value.sum(axis=-1, keepdims=True)


def test_equations_match_independent_numpy_in_autocast():
    rng = np.random.default_rng(5)
    visual, text = rng.normal(size=(12, 8)), rng.normal(size=(5, 8))
    q, k = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 18, 4))
    ids = np.arange(1, 13)
    expected_alignment = softmax(visual @ text.T).mean(axis=0)
    expected_anchors = np.flatnonzero(expected_alignment > expected_alignment.mean())
    expected_score = softmax(q @ k.transpose(0, 2, 1) / 2).mean(axis=(0, 1))[ids]

    def tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        anchors = text_anchors(tensor(visual), tensor(text))
        score = semantic_scores(tensor(q), tensor(k), torch.tensor(ids))
    np.testing.assert_array_equal(anchors.numpy(), expected_anchors)
    np.testing.assert_allclose(score.numpy(), expected_score, rtol=2e-6, atol=1e-7)
    assert score.dtype == torch.float32


def test_score_normalizes_over_nonvisual_keys_too():
    q = torch.ones(2, 1, 2)
    k = torch.zeros(2, 5, 2)
    score = semantic_scores(q, k, torch.tensor([0, 1]))
    torch.testing.assert_close(score, torch.tensor([0.2, 0.2]))
    k[:, -1] = 10
    assert bool((semantic_scores(q, k, torch.tensor([0, 1])) < score / 100).all())


@pytest.mark.parametrize("views", [[0] * 10, [0] * 5 + [1] * 5])
def test_diversity_uses_full_current_view_independent_of_semantic_rejects(views):
    rng = np.random.default_rng(9)
    h = rng.normal(size=(10, 6)).astype(np.float32)
    ids = np.array(views)
    expected = []
    unit = h / np.linalg.norm(h, axis=1, keepdims=True)
    for i in range(len(h)):
        others = [j for j in range(len(h)) if j != i and ids[j] == ids[i]]
        expected.append(1 - sum(np.dot(unit[i], unit[j]) for j in others) / len(others))
    actual = view_diversity(torch.tensor(h), torch.tensor(ids))
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7)
    scores = np.linspace(0, 1, 10, dtype=np.float32)
    selected, stats = select_tokens(
        torch.tensor(h), torch.tensor(scores), torch.tensor(ids), 0.6, 0.34
    )
    semantic = np.argsort(-scores, kind="stable")[:4].tolist()
    candidates = [i for i in range(10) if i not in semantic]
    recycled = sorted(candidates, key=lambda i: (-expected[i], i))[:2]
    np.testing.assert_array_equal(selected.nonzero().flatten().numpy(), sorted(semantic + recycled))
    assert stats == dict(input_tokens=10, output_tokens=6, semantic_tokens=4, recycled_tokens=2)


def test_degenerate_features_stable_ties_and_both_components():
    h = torch.zeros(6, 8)
    assert text_anchors(h, torch.zeros(3, 8)).numel() == 0
    assert text_anchors(h, torch.ones(1, 8)).numel() == 0
    selected, stats = select_tokens(h, torch.zeros(6), torch.tensor([0, 0, 0, 0, 0, 1]), 0.01, 0.01)
    assert selected.nonzero().flatten().tolist() == [0, 1]
    assert stats["semantic_tokens"] == stats["recycled_tokens"] == 1
    assert view_diversity(h, torch.arange(6)).tolist() == [0.0] * 6


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"keep_ratio": 0},
        {"keep_ratio": float("nan")},
        {"keep_ratio": float("inf")},
        {"keep_ratio": True},
        {"recycle_ratio": 0},
        {"recycle_ratio": 1},
        {"pruning_layer": -1},
        {"pruning_layer": 2.5},
        {"pruning_layer": True},
        {"unknown": 1},
    ],
)
def test_configuration_rejects_invalid_or_disabled_core_parameters(parameters):
    with pytest.raises(ValueError):
        ILSAConfig.parse(parameters)


def test_strategy_protects_rows_defers_selection_and_clears_state():
    torch.manual_seed(11)

    class Strategy(ILSAStrategy):
        def language_mask(self, state):
            return state.positions >= 7

    layer = SimpleNamespace(
        input_layernorm=torch.nn.Identity(),
        self_attn=SimpleNamespace(
            q_proj=torch.nn.Linear(8, 8, bias=False),
            k_proj=torch.nn.Linear(8, 8, bias=False),
            head_dim=4,
            num_heads=2,
            num_key_value_heads=2,
        ),
    )
    backend = SimpleNamespace(query_diagnostics={})
    method = Strategy(
        backend, ILSAConfig(pruning_layer=0, keep_ratio=0.5), SimpleNamespace(layers=[layer, layer])
    )
    layout = SimpleNamespace(patches_per_view=3, visual_mask=lambda p: (p >= 1) & (p < 7))
    state = SimpleNamespace(positions=torch.arange(11), hidden=torch.randn(1, 11, 8), layout=layout)
    method.before_query(None)
    assert method.before_layer(0, state) is None
    keep = method.selection_after_layer(0)
    assert keep is not None and bool(keep[[0, 7, 8, 9, 10]].all())
    assert int(keep[1:7].sum()) == 3
    assert method.pending is None
    state.positions = state.positions[keep]
    method.finish(state)
    assert backend.query_diagnostics["visual_tokens_final"] == 3
    method.reset(None)
    assert method.pending is None and method.stage == {}
    method.close()
    method.close()


def test_empty_anchor_fallback_is_explicit_and_does_not_make_fake_scores():
    class Strategy(ILSAStrategy):
        def language_mask(self, state):
            return state.positions == 4

    layer = SimpleNamespace(input_layernorm=torch.nn.Identity())
    method = Strategy(
        SimpleNamespace(query_diagnostics={}),
        ILSAConfig(pruning_layer=0),
        SimpleNamespace(layers=[layer, layer]),
    )
    state = SimpleNamespace(
        positions=torch.arange(5),
        hidden=torch.zeros(1, 5, 4),
        layout=SimpleNamespace(visual_mask=lambda p: (p > 0) & (p < 4)),
    )
    method.before_query(None)
    method.before_layer(0, state)
    assert method.selection_after_layer(0) is None
    assert method.stage["empty_anchor_dense_fallback"] is True
    method.abort()
    assert method.pending is None
