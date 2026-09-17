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
