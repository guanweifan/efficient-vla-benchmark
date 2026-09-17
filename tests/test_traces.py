import numpy as np
import pytest

from efficient_vla_benchmark.core.types import EpisodeContext, Observation
from efficient_vla_benchmark.evaluation.libero import make_observation
from efficient_vla_benchmark.evaluation.traces import load_trace, save_trace


def test_trace_round_trip_without_pickle(tmp_path):
    obs = Observation(
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.ones((16, 16, 3), dtype=np.uint8),
        np.arange(8, dtype=np.float32),
        "pick up the object",
    )
    ctx = EpisodeContext("libero_spatial", 0, 0, 7)
    path = tmp_path / "trace.npz"
    save_trace(path, ctx, [obs, obs], [0, 1])
    restored, samples, steps = load_trace(path, require_control_steps=True)
    assert restored == ctx
    assert len(samples) == 2 and samples[0].instruction == obs.instruction
    np.testing.assert_array_equal(samples[0].image, obs.image)
    np.testing.assert_array_equal(samples[0].proprio, obs.proprio)
    np.testing.assert_array_equal(steps, [0, 1])


def test_sparse_query_trace_cannot_impersonate_control_trace(tmp_path):
    obs = Observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(8),
        "test",
    )
    path = tmp_path / "trace.npz"
    save_trace(path, EpisodeContext("libero_goal", 0, 0), [obs, obs], [0, 8], sampling="query")
    assert len(load_trace(path)[1]) == 2
    with pytest.raises(ValueError, match="all control frames"):
        load_trace(path, require_control_steps=True)


def test_trace_rejects_nonsequential_steps(tmp_path):
    obs = Observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(8),
        "test",
    )
    path = tmp_path / "trace.npz"
    save_trace(path, EpisodeContext("libero_goal", 0, 0), [obs, obs], [8, 0])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_trace(path)


def test_libero_orientation_and_proprio_do_not_mutate_input():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    quaternion = np.array([0.0, 0.0, 0.0, 1.000001])
    raw = {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": np.ones(3),
        "robot0_eef_quat": quaternion,
        "robot0_gripper_qpos": np.zeros(2),
    }
    obs = make_observation(raw, "test")
    np.testing.assert_array_equal(obs.image, image[::-1, ::-1])
    np.testing.assert_array_equal(obs.proprio[3:6], np.zeros(3))
    assert quaternion[3] == 1.000001
