"""Metric conversion, orientation and explicit depth trace contracts."""

from dataclasses import replace

import numpy as np
import pytest

from efficient_vla_benchmark.core.types import EpisodeContext, Observation
from efficient_vla_benchmark.evaluation.libero.observations import make_observation
from efficient_vla_benchmark.evaluation.traces import load_trace, save_trace


def sample():
    return Observation(
        np.zeros((2, 2, 3), np.uint8),
        np.zeros((2, 2, 3), np.uint8),
        np.zeros(8),
        "test",
        np.ones((2, 2), np.float32),
    )


def test_depth_metric_conversion_and_shared_rgb_orientation():
    buffer = np.array([[[0.0], [0.25]], [[0.5], [1.0]]])
    raw = dict(
        agentview_image=sample().image,
        robot0_eye_in_hand_image=sample().image,
        robot0_eef_pos=np.zeros(3),
        robot0_eef_quat=np.array([0, 0, 0, 1]),
        robot0_gripper_qpos=np.zeros(2),
        agentview_depth=buffer,
    )
    observation = make_observation(raw, "test", (0.1, 10.0))
    expected = (1 / (10 - 9.9 * buffer[..., 0]))[::-1, ::-1]
    np.testing.assert_allclose(observation.depth, expected, rtol=1e-6)
    np.testing.assert_array_equal(raw["agentview_depth"], buffer)
    observation.validate()
    with pytest.raises(ValueError, match="near/far"):
        make_observation(raw, "test")


def test_schema3_roundtrip_does_not_fabricate_depth_for_old_traces(tmp_path):
    context = EpisodeContext("libero_goal", 0, 0)
    for observation in (sample(), replace(sample(), depth=None)):
        path = tmp_path / ("depth.npz" if observation.depth is not None else "rgb.npz")
        save_trace(path, context, [observation], [0])
        restored = load_trace(path, require_control_steps=True)[1][0]
        if observation.depth is None:
            assert restored.depth is None
        else:
            np.testing.assert_array_equal(restored.depth, observation.depth)
    with pytest.raises(ValueError, match="mix"):
        save_trace(
            tmp_path / "mixed.npz", context, [sample(), replace(sample(), depth=None)], [0, 1]
        )


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((2, 2), np.float32),
        np.ones((2, 2)),
        np.ones((3, 2), np.float32),
        np.full((2, 2), np.nan, np.float32),
    ],
)
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="Depth"):
        replace(sample(), depth=depth).validate()
