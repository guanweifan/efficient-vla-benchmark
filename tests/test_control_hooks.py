from dataclasses import replace

import numpy as np
import pytest

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.lifecycle import MethodHost
from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.types import EpisodeContext, Observation


def observation():
    return Observation(
        np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), np.zeros(8), "test"
    )


class Host(MethodHost):
    def __init__(self, method):
        self.initialize_method_state()
        self.attach_method(method)
        self.calls = 0

    def reset(self, context):
        self.method.reset(context)

    def predict_chunk(self, observation):
        self.calls += 1
        return np.arange(28, dtype=float).reshape(4, 7)


class Hook(InferenceMethod):
    def reset(self, context):
        self.plans, self.actions = [], []

    def before_plan(self, observation, step):
        return np.full(7, 42.0) if step == 2 else None

    def on_plan(self, actions, step):
        self.plans.append((step, actions.copy()))
        actions[:] = -100

    def transform_action(self, action, step):
        return action + 1

    def after_action(self, action, remaining):
        self.actions.append(action.copy())
        action[:] = -100
        return False


def test_full_chunk_single_replacement_and_actual_queries():
    method = Hook()
    host = Host(method)
    policy = ChunkPolicy(host, 2)
    policy.reset(EpisodeContext("test", 0, 0))
    decisions = [policy.step(observation()) for _ in range(5)]
    assert [d.queried for d in decisions] == [True, False, False, True, False]
    assert policy.query_count == host.calls == 2
    assert [s for s, _ in method.plans] == [0, 3]
    assert all(a.shape == (4, 7) for _, a in method.plans)
    np.testing.assert_array_equal(decisions[0].action, np.arange(7) + 1)
    np.testing.assert_array_equal(decisions[2].action, np.full(7, 43.0))
    assert decisions[2].diagnostics["action_source"] == "replacement"
    np.testing.assert_array_equal(method.actions[2], decisions[2].action)
    policy.reset(EpisodeContext("test", 0, 1))
    assert method.plans == [] and method.actions == []


def test_instruction_change_drops_pending_actions_and_method_history():
    host = Host(Hook())
    policy = ChunkPolicy(host, 4)
    policy.reset(EpisodeContext("test", 0, 0))
    policy.step(observation())
    assert policy.step(replace(observation(), instruction="new task")).queried
    assert len(host.method.plans) == 1


@pytest.mark.parametrize("bad", [np.zeros((2, 7)), np.zeros(6), np.full(7, np.nan)])
def test_invalid_replacement_fails_before_emission(bad):
    host = Host(Hook())
    policy = ChunkPolicy(host, 1)
    policy.reset(EpisodeContext("test", 0, 0))
    policy.step(observation())
    host.method.before_plan = lambda obs, step: bad
    with pytest.raises(ValueError, match="one finite action"):
        policy.step(observation())
