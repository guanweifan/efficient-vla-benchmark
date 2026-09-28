"""Independent pre-projection equations and native sparse projection integration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import VisionZipConfig
from ..grouping import compress_view, visual_evidence
from ..strategy import VisionZipStrategy


def numpy_evidence(query, key, scale):
    scores = np.einsum("bhid,bhjd->bhij", query.astype(np.float64), key.astype(np.float64)) * scale
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return weights.mean((1, 2))[0], key.astype(np.float64).mean(1)[0]


def numpy_reference(features, scores, metric, target, ratio):
    features = features.astype(np.float64)
    metric = metric.astype(np.float64)
    metric /= np.maximum(np.linalg.norm(metric, axis=1, keepdims=True), 1e-12)
    dominant_count = min(target - 1, max(1, int(target * ratio)))
    dominant = sorted(
        sorted(range(len(features)), key=lambda i: (-float(scores[i]), i))[:dominant_count]
    )
    residual = [i for i in range(len(features)) if i not in dominant]
    contextual_count = target - dominant_count
    seeds = residual[:: max(1, len(residual) // contextual_count)][:contextual_count]
    groups = {i: [] for i in seeds}
    for index in residual:
        if index in seeds:
            continue
        similarities = [float(metric[index] @ metric[seed]) for seed in seeds]
        groups[seeds[int(np.argmax(similarities))]].append(index)
    rows = [(index, features[index]) for index in dominant]
    for seed, members in groups.items():
        aggregate = features[members].mean(0) if members else np.zeros(features.shape[1])
        rows.append((max([seed] + members), features[seed] + aggregate))
    rows.sort(key=lambda x: x[0])
    return np.stack([r[1] for r in rows]), np.asarray([r[0] for r in rows])


@pytest.mark.parametrize("count,target", [(2, 2), (7, 2), (7, 7), (31, 8), (64, 16), (256, 128)])
def test_independent_grouping_and_released_aggregation(count, target):
    for seed in range(5):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(count, 13)).astype(np.float32)
        metric = rng.normal(size=(count, 8)).astype(np.float32)
        scores = rng.uniform(size=count).astype(np.float32)
        expected, anchors = numpy_reference(features, scores, metric, target, 0.84375)
        actual, actual_anchors = compress_view(
            torch.from_numpy(features),
            torch.from_numpy(scores),
            torch.from_numpy(metric),
            target,
            0.84375,
        )
        np.testing.assert_array_equal(actual_anchors.numpy(), anchors)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=3e-7)
        assert len(set(anchors)) == target and np.all(np.diff(anchors) > 0)


def test_received_attention_and_mean_head_keys_not_outgoing_attention():
    torch.manual_seed(19)
    query, key = torch.randn(1, 3, 16, 8), torch.randn(1, 3, 16, 8)
    scores, metric = visual_evidence(query, key, 0.35)
    expected_scores, expected_metric = numpy_evidence(query.numpy(), key.numpy(), 0.35)
    np.testing.assert_allclose(scores.numpy(), expected_scores, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(metric.numpy(), expected_metric, rtol=2e-6, atol=1e-7)
    assert scores.std() > 0.001


def test_released_target_plus_other_mean_is_not_pooled_average():
    features = torch.arange(1, 6, dtype=torch.float32).reshape(-1, 1)
    scores = torch.tensor([9.0, 1.0, 2.0, 3.0, 4.0])
    merged, anchors = compress_view(features, scores, torch.ones(5, 2), 2, 0.5)
    # Dominant row0; uniform residual target row1; rows2/3/4 join it.
    assert anchors.tolist() == [0, 4]
    assert merged.flatten().tolist() == [1.0, 6.0]
    assert merged[-1].item() != features[1:].mean().item()


def test_autocast_does_not_change_fp32_equations():
    torch.manual_seed(87)
    query, key = torch.randn(1, 3, 32, 8), torch.randn(1, 3, 32, 8)
    features = torch.randn(32, 12)
    scores, metric = numpy_evidence(query.numpy(), key.numpy(), 0.35)
    expected, anchors = numpy_reference(features.numpy(), scores, metric, 8, 0.75)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual_scores, actual_metric = visual_evidence(query, key, 0.35)
        actual, actual_anchors = compress_view(features, actual_scores, actual_metric, 8, 0.75)
    assert actual_scores.dtype == actual_metric.dtype == actual.dtype == torch.float32
    np.testing.assert_array_equal(actual_anchors.numpy(), anchors)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=3e-7)


def test_zero_norms_and_empty_context_groups_remain_finite():
    for count, target in ((7, 2), (7, 7)):
        merged, anchors = compress_view(
            torch.zeros(count, 3), torch.ones(count), torch.zeros(count, 2), target, 0.5
        )
        assert torch.isfinite(merged).all() and len(set(anchors.tolist())) == target


@pytest.mark.parametrize("target", [0, 1, 5, True, 2.5])
def test_invalid_target(target):
    with pytest.raises(ValueError, match="target"):
        compress_view(torch.ones(4, 3), torch.ones(4), torch.ones(4, 2), target, 0.5)


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"keep_ratio": True},
        {"keep_ratio": 0},
        {"keep_ratio": 1.1},
        {"dominant_ratio": 0},
        {"dominant_ratio": 1},
        {"dominant_ratio": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        VisionZipConfig.parse(parameters)


class SquareProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, features):
        self.inputs.append(features.detach().clone())
        return features.square() + 0.25


def setup_method(views=2, ratio=0.5):
    block = SimpleNamespace(
        attn=SimpleNamespace(q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(), scale=0.5)
    )
    encoder = SimpleNamespace(
        training=False,
        cls_token=None,
        num_prefix_tokens=0,
        patch_embed=SimpleNamespace(num_patches=4),
        blocks=[block, block],
    )
    projector = SquareProjector().eval()
    backend = SimpleNamespace(query_diagnostics={})
    method = VisionZipStrategy(backend, VisionZipConfig(keep_ratio=ratio), encoder, projector)
    layout = SimpleNamespace(
        views=views, patches_per_view=4, visual_mask=lambda p: (p >= 1) & (p < 1 + views * 4)
    )
    state = SimpleNamespace(layout=layout, positions=torch.arange(views * 4 + 3))
    return method, encoder, projector, state


def feed_evidence(encoder, views):
    pairs = []
    for _ in range(views):
        query, key = torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)
        encoder.blocks[0].attn.q_norm(query)
        encoder.blocks[0].attn.k_norm(key)
        pairs.append((query, key))
    return pairs


def test_sparse_projection_occurs_after_fused_merge_and_before_llm_selection():
    method, encoder, projector, state = setup_method()
    native_forward = projector.forward
    method.before_query({})
    torch.manual_seed(71)
    features = torch.randn(1, 8, 7)
    pairs = feed_evidence(encoder, 2)
    expected, positions = [], []
    for view, (q, k) in enumerate(pairs):
        score, metric = numpy_evidence(q.numpy(), k.numpy(), 0.5)
        values, anchors = numpy_reference(
            features[0, view * 4 : (view + 1) * 4].numpy(), score, metric, 2, 0.84375
        )
        expected.append(torch.tensor(values, dtype=features.dtype))
        positions.append(torch.from_numpy(anchors) + view * 4)
    with torch.no_grad():
        output = projector(features)
    compact = torch.cat(expected)
    anchors = torch.cat(positions)
    torch.testing.assert_close(projector.inputs[-1][0], compact, rtol=1e-5, atol=1e-6)
    assert projector.inputs[-1].shape == (1, 4, 7)
    torch.testing.assert_close(output[0, anchors], compact.square() + 0.25, rtol=1e-5, atol=1e-6)
    empty = torch.ones(8, dtype=torch.bool)
    empty[anchors] = False
    assert output[0, empty].count_nonzero() == 0
    method.start(state)
    keep = method.before_layer(0, state)
    assert keep[[0, 9, 10]].all()
    assert torch.equal(state.positions[keep][1:5], anchors + 1)
    method.finish(state)
    assert method.backend.query_diagnostics["visionzip_projector_input_tokens"] == 4
    assert method.probe.take() == [] and method.anchors is None
    with pytest.raises(RuntimeError, match="missing or repeated"):
        method.before_layer(0, state)
    method.close()
    method.close()
    assert "forward" not in projector.__dict__
    assert not hasattr(projector, "_evb_projector_compression")
    with torch.no_grad():
        torch.testing.assert_close(projector(features), features.square() + 0.25)
    assert native_forward != projector.forward  # It was the method wrapper, now removed.
    with pytest.raises(RuntimeError, match="closed"):
        method.before_query({})


def test_errors_clear_evidence_and_repeated_native_projection_is_rejected():
    method, encoder, projector, state = setup_method()
    try:
        method.before_query({})
        with torch.no_grad(), pytest.raises(RuntimeError, match="one native SigLIP"):
            projector(torch.ones(1, 8, 3))
        assert not method.query_open and method.probe.take() == []
        method.before_query({})
        feed_evidence(encoder, 2)
        with torch.no_grad():
            projector(torch.ones(1, 8, 3))
            with pytest.raises(RuntimeError, match="one native inference"):
                projector(torch.ones(1, 8, 3))
        method.abort()
        assert not method.compressed and method.anchors is None
    finally:
        method.close()


def test_full_retention_and_unsupported_prefix_are_explicit():
    method, encoder, projector, state = setup_method(ratio=1)
    assert method.probe is None and "forward" not in projector.__dict__
    method.close()
    for changes in ({"cls_token": torch.ones(1)}, {"num_prefix_tokens": 1}, {"training": True}):
        invalid = SimpleNamespace(**(vars(encoder) | changes))
        with pytest.raises(ValueError, match="non-CLS"):
            VisionZipStrategy(SimpleNamespace(), VisionZipConfig(), invalid, projector)
