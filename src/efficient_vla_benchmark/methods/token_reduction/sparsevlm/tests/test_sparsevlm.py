import numpy as np
import pytest
import torch

from ..config import SparseVLMConfig
from ..recycling import density_clusters, prune_and_recycle, text_raters


def test_raters_match_independent_numpy_softmax():
    visual = np.array([[1.0, 2.0], [2.0, -1.0]])
    text = np.array([[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]])
    logits = visual @ text.T
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    relevance = (weights / weights.sum(-1, keepdims=True)).mean(0)
    np.testing.assert_array_equal(
        text_raters(torch.tensor(visual), torch.tensor(text)), relevance >= relevance.mean()
    )


def test_rank_controls_budget_and_reconstruction_sums():
    features = torch.arange(32.0).reshape(8, 4) + 1
    p = torch.arange(1.0, 9.0)[None].repeat(2, 1)
    config = SparseVLMConfig(rank_scale=1, recycle_ratio=1, cluster_ratio=0.2)
    ids, values, stats = prune_and_recycle(features, p, torch.arange(8), config)
    assert stats == {"rank": 1, "removed": 7, "recycled": 7, "clusters": 1}
    # Most salient row survives; the other seven sum at their latest anchor.
    assert ids.tolist() == [6, 7]
    torch.testing.assert_close(values, torch.stack((features[:7].sum(0), features[7])))


def test_full_rank_skips_sparsification_and_large_density_is_finite():
    features = torch.randn(8, 16) * 100
    ids, result, stats = prune_and_recycle(
        features, torch.eye(8), torch.arange(8), SparseVLMConfig()
    )
    assert ids.tolist() == list(range(8)) and stats["removed"] == 0
    torch.testing.assert_close(result, features)
    assignment = density_clusters(features, 3, 3)
    assert sorted(assignment.unique().tolist()) == [0, 1, 2]
    assert torch.equal(assignment, density_clusters(features, 3, 3))


def test_degenerate_clusters_preserve_centers_and_global_rng():
    state = torch.get_rng_state()
    assignment = density_clusters(torch.zeros(7, 8), 3, 3)
    assert sorted(assignment.unique().tolist()) == [0, 1, 2]
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize(
    "parameters",
    [
        {"pruning_layers": [3, 2]},
        {"pruning_layers": []},
        {"rank_scale": 0},
        {"recycle_ratio": True},
        {"neighbors": 0},
        {"rank_rtol": float("nan")},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        SparseVLMConfig.parse(parameters)


@pytest.mark.gpu
def test_real_oft_recycling_reset_and_detach(model_path, trace_path, request):
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
    assert len(stages) == 3 and all(stage["clusters"] > 0 for stage in stages)
    assert backend.method.raters is None
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
