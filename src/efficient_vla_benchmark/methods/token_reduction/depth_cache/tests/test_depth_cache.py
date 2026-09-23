import pytest
import torch

from ..config import DepthCacheConfig
from ..merging import ProgressiveDepthMerge, region_partition


def test_partition_protects_tokens_and_separates_depth():
    depth = torch.tensor([1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 8.0, 8.0, 8.0])
    protected = torch.zeros(9, dtype=torch.bool)
    protected[0] = True
    result = region_partition(depth, protected, 3)
    assert result[0] == -1
    assert result[1] == result[2] and result[3] == result[5] and result[6] == result[8]
    assert len(result[1:].unique()) == 3


def test_progressive_budget_current_features_protection_and_dynamic_recovery():
    config = DepthCacheConfig(
        warmup_queries=1,
        merge_window=2,
        static_threshold_m=0.1,
        dynamic_fraction=0.2,
        min_merge_ratio=0.5,
        max_merge_ratio=0.7,
    )
    core = ProgressiveDepthMerge(config)
    depth = torch.tensor([1.0] * 4 + [3.0] * 4 + [8.0] * 4)
    edges = torch.zeros(12, dtype=torch.bool)
    edges[0] = True
    features = torch.arange(48.0).reshape(12, 4)
    core.observe_depth(depth, edges)
    core.collect_saliency(torch.ones(12))
    counts = []
    for _ in range(4):
        core.observe_depth(depth, edges)
        anchors, merged = core.reduce(features)
        counts.append(len(anchors))
    assert counts[0] == 12 and counts[-1] < counts[0]
    assert counts == sorted(counts, reverse=True)
    assert 0 in anchors.tolist()
    # Fresh features, not cached values, determine the weighted reconstruction.
    _, doubled = core.reduce(features * 2)
    torch.testing.assert_close(doubled, merged * 2)
    changed = depth.clone()
    changed[-4:] += 1
    core.observe_depth(changed, edges)
    recovered, _ = core.reduce(features)
    assert all(index in recovered.tolist() for index in range(8, 12))
    assert core.dynamic_regions > 0
    core.reset()
    assert core.warming and core.groups is None and not core.depth_history


@pytest.mark.parametrize(
    "params",
    [
        {"warmup_queries": 0},
        {"max_merge_ratio": 1},
        {"edge_threshold_m": 0},
        {"wrist_gating": True},
    ],
)
def test_invalid_config(params):
    with pytest.raises(ValueError):
        DepthCacheConfig.parse(params)
