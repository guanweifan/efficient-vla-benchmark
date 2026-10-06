import numpy as np
import pytest
import torch

from ..config import SparseVLMConfig
from ..selection import REFERENCE_BUDGETS, select_visual, stage_budget, text_raters


def test_raters_follow_author_strict_threshold():
    visual = np.array([[1.0, 2.0], [2.0, -1.0]], np.float32)
    text = np.array([[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]], np.float32)
    logits = visual @ text.T
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    relevance = (weights / weights.sum(-1, keepdims=True)).mean(0)
    np.testing.assert_array_equal(
        text_raters(torch.tensor(visual), torch.tensor(text)), relevance > relevance.mean()
    )
    with pytest.raises(ValueError, match="no text raters"):
        text_raters(torch.zeros(8, 4), torch.zeros(3, 4))


@pytest.mark.parametrize("profile", [192, 128, 96, 64])
def test_official_budgets_and_native_patch_mapping(profile):
    config = SparseVLMConfig(budget_profile=profile)
    assert tuple(stage_budget(576, i, config) for i in range(3)) == REFERENCE_BUDGETS[profile]
    assert tuple(stage_budget(256, i, config) for i in range(3)) == tuple(
        max(1, int(value * 256 / 576)) for value in REFERENCE_BUDGETS[profile]
    )


def test_priority_heads_and_topk_match_author_operations():
    torch.manual_seed(65)
    p = torch.rand(32, 7, 44)
    visual = torch.zeros(44, dtype=torch.bool)
    visual[1:33] = True
    # Full reference sequence: author select_attn_head_by_sum, then head/rater means.
    heads = p[:, :, 1:33].sum((1, 2)).topk(14).indices
    score = p[heads].mean(0)[:, 1:33].mean(0)
    expected = score.topk(12).indices.sort().values
    ids, stats = select_visual(p, visual, 12, 14)
    assert torch.equal(ids, expected)
    assert stats == {"kept": 12, "removed": 20, "priority_heads": 14}
    assert select_visual(p, visual, 100, 14)[1]["kept"] == 31
    with pytest.raises(ValueError, match="Nonfinite"):
        select_visual(p.fill_(float("nan")), visual, 12, 14)


@pytest.mark.parametrize(
    "parameters",
    [
        {"pruning_layers": [3, 2, 6]},
        {"pruning_layers": []},
        {"budget_profile": 100},
        {"budget_scale": 0},
        {"budget_scale": True},
        {"priority_heads": 0},
        {"rank_scale": 0.5},
        {"recycle_ratio": 0.3},
    ],
)
def test_invalid_or_retired_config(parameters):
    with pytest.raises(ValueError):
        SparseVLMConfig.parse(parameters)


@pytest.mark.gpu
def test_real_oft_v2_selection_reset_and_detach(model_path, trace_path, request):
    pytest.importorskip("efficient_vla_backend_openvla_oft")
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import SparseVLM

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    backend.attach_method(SparseVLM(backend, SparseVLMConfig(enabled=False)))
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
    backend.detach_method()
    backend.attach_method(SparseVLM(backend, SparseVLMConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(observations[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert 0 < backend.query_diagnostics["visual_tokens_final"] < 512
    stages = backend.query_diagnostics["sparsevlm_stages"]
    assert len(stages) == 3 and all(stage["removed"] > 0 for stage in stages)
    assert backend.method.raters is None
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
