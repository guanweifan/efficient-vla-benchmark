import numpy as np
import torch

from efficient_vla_benchmark.model_ops.depth import depth_patch_signals


def test_depth_geometry_keeps_metric_units_edges_and_crop_center():
    depth = np.ones((224, 224), np.float32)
    depth[:, 112:] = 2
    for crop in (1.0, 0.9):
        values, edges = depth_patch_signals(depth, "cpu", crop, 0.1)
        assert values.shape == edges.shape == (256,)
        assert values.min() == 1 and values.max() == 2
        grid = edges.reshape(16, 16)
        assert grid[:, 7:9].all() and not grid[:, :7].any() and not grid[:, 9:].any()
        assert torch.equal(values.reshape(16, 16)[0], values.reshape(16, 16)[-1])
