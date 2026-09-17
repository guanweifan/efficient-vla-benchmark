import numpy as np
import pytest

from efficient_vla_benchmark.core.types import ActionDecision, EpisodeContext
from efficient_vla_benchmark.evaluation.libero import SUITES, run_episode, validate_protocol


def test_smoke_and_explicit_formal_protocol():
    validate_protocol(list(SUITES), [0], 1, False)
    validate_protocol(list(SUITES), list(range(10)), 50, True)


@pytest.mark.parametrize(
    "suites,tasks,trials",
    [
        ([], [0], 1),
        (["unknown"], [0], 1),
        (["libero_goal", "libero_goal"], [0], 1),
        (["libero_goal"], [0, 0], 1),
        (["libero_goal"], [], 1),
        (["libero_goal"], [10], 1),
        (["libero_goal"], [True], 1),
        (["libero_goal"], [0], 0),
        (["libero_goal"], [0], 51),
        (list(SUITES), [0], 2),
    ],
)
def test_invalid_protocol_rejected_without_loading_simulator(suites, tasks, trials):
    with pytest.raises(ValueError):
        validate_protocol(suites, tasks, trials, False)


def test_rollout_steps_every_action_and_records_nonquery_frames():
    class Environment:
        calls = 0

        def seed(self, value):
            assert value == 0

        def reset(self):
            self.calls = 0

        def set_init_state(self, state):
            return {
                "robot0_eef_quat": np.array([0, 0, 0, 1]),
                "robot0_eef_pos": np.zeros(3),
                "robot0_gripper_qpos": np.zeros(2),
                "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            }

        def step(self, action):
            self.calls += 1
            return self.set_init_state(None), 0, self.calls == 13, {}

    class Policy:
        def reset(self, context):
            self.steps = 0

        def step(self, obs):
            self.steps += 1
            return ActionDecision(np.zeros(7), self.steps == 1)

    environment, policy = Environment(), Policy()
    result, samples, steps = run_episode(
        environment, policy, EpisodeContext("libero_goal", 0, 0), None, "test", True
    )
    assert environment.calls == 13  # Ten settling steps, then three policy actions.
    assert result["success"] and result["steps"] == 3 and result["policy_queries"] == 1
    assert len(samples) == 3 and steps == [0, 1, 2]
