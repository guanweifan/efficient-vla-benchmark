import numpy as np
import pytest
import torch

from ..config import STMergeConfig
from ..implementations.openvla_oft.method import spatial_groups


def test_grouping_retains_anchors_and_complete_membership():
    torch.manual_seed(73)
    keys = torch.randn(256, 32)
    reduction = spatial_groups(keys, 128, 4, 2)
    assert 128 <= reduction.anchors.numel() < 256
    assert reduction.groups.shape == (256,)
    assert (reduction.groups >= 0).all()
    assert (reduction.groups < reduction.anchors.numel()).all()
    assert set(range(0, 256, 4)) <= set(reduction.anchors.tolist())
    torch.testing.assert_close(
        reduction.groups[reduction.anchors], torch.arange(reduction.anchors.numel())
    )
    again = spatial_groups(keys, 128, 4, 2)
    assert torch.equal(again.groups, reduction.groups)


def test_full_retention_is_identity():
    result = spatial_groups(torch.zeros(256, 4), 256, 4, 2)
    assert result.groups.tolist() == list(range(256))


def test_dot_product_preserves_key_magnitude_in_matching():
    # Anchor 0 prefers candidate 3 by Eq. 2, but cosine would pick candidate 1.
    keys = torch.tensor([[1.0, 0], [1.0, 0], [0.0, 1], [10.0, 1]])
    result = spatial_groups(keys, target=3, queues=2, sigma=100, side=2)
    assert result.anchors.tolist() == [0, 1, 2]
    assert result.groups.tolist() == [0, 1, 2, 0]


def test_spatial_groups_match_independent_scalar_dot_product_reference():
    import math

    rng = np.random.default_rng(14)
    keys = rng.normal(size=(16, 7)) * rng.uniform(0.2, 4, size=(16, 1))
    proposals = []
    for queue in range(1, 4):
        for anchor in range(0, 16, 4):
            candidates = list(range(queue, 16, 4))
            scores = []
            for candidate in candidates:
                distance = (anchor // 4 - candidate // 4) ** 2 + (anchor % 4 - candidate % 4) ** 2
                scores.append(float(keys[anchor] @ keys[candidate]) * math.exp(-distance / 8))
            best = int(np.argmax(scores))
            proposals.append((scores[best], candidates[best], anchor))
    representatives = list(range(16))
    used = set()
    for _, source, anchor in sorted(proposals, key=lambda item: -item[0]):
        if source not in used and len(used) < 8:
            representatives[source] = anchor
            used.add(source)
    kept = [i for i, representative in enumerate(representatives) if i == representative]
    result = spatial_groups(torch.tensor(keys), 8, 4, 2, side=4)
    assert result.anchors.tolist() == kept
    assert result.groups.tolist() == [kept.index(i) for i in representatives]


@pytest.mark.parametrize(
    "parameters",
    [
        {"queues": 1},
        {"tokens_per_view": 2},
        {"spatial_sigma": 0},
        {"merging_layer": 22},
        {"enabled": 1},
    ],
)
def test_bad_config(parameters):
    with pytest.raises(ValueError):
        STMergeConfig.parse(parameters)


@pytest.mark.gpu
def test_real_oft_noop_merge_reset_restore(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import STMerge

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for params in ({"enabled": False}, {"tokens_per_view": 256}):
        backend.attach_method(STMerge(backend, STMergeConfig.parse(params)))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    backend.attach_method(STMerge(backend, STMergeConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(samples[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert 256 <= backend.query_diagnostics["visual_tokens_final"] < 512
    for counts in backend.query_diagnostics["vision_patch_counts"].values():
        assert counts[0] == counts[1] == 256 and counts[2] < 256
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), actual)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
