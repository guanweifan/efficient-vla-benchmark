"""Dynamic prefixes stay separate from model length and native reference cadence."""

from dataclasses import replace

import numpy as np
import pytest

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.lifecycle import MethodHost
from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.types import EpisodeContext, Observation


def observation():
    return Observation(
        np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), np.zeros(8), "task"
    )


class Adaptive(InferenceMethod):
    def __init__(self, horizon):
        self.horizon = horizon

    def reset(self, context):
        self.planned = False
        self.seen = []

    def on_plan(self, actions, step):
        self.planned = True
        actions[:] = -100

    def select_execution_horizon(self, actions, default_horizon):
        assert self.planned
        self.seen.append((actions.copy(), default_horizon))
        actions[:] = -200
        return self.horizon


class Host(MethodHost):
    def __init__(self, method=None, length=16):
        self.initialize_method_state()
        if method is not None:
            self.attach_method(method)
        self.length = length
        self.calls = 0

    def reset(self, context):
        if self.method:
            self.method.reset(context)

    def predict_chunk(self, observation):
        self.calls += 1
        return np.arange(self.length * 7).reshape(self.length, 7)


def policy_for(method=None, length=16):
    host = Host(method, length)
    policy = ChunkPolicy(host, 8)
    policy.reset(EpisodeContext("suite", 0, 0))
    return policy, host


@pytest.mark.parametrize("horizon", [1, 3, 8, 12, 16])
def test_real_prefix_query_count_and_mutation_isolation(horizon):
    method = Adaptive(horizon)
    policy, host = policy_for(method)
    decisions = [policy.step(observation()) for _ in range(horizon + 1)]
    assert [i for i, d in enumerate(decisions) if d.queried] == [0, horizon]
    assert policy.query_count == host.calls == 2
    assert policy.execution_horizon == 8
    for i, decision in enumerate(decisions):
        np.testing.assert_array_equal(decision.action, np.arange(7) + 7 * (i % horizon))
        assert decision.diagnostics["selected_execution_horizon"] == horizon
    assert method.seen[0][0].shape == (16, 7)
    assert method.seen[0][1] == 8
    assert method.seen[0][0][0, 0] == 0


@pytest.mark.parametrize("horizon", [0, -1, 17, 1.5, True, None, np.int64(4)])
def test_invalid_horizon_never_emits_or_queues_actions(horizon):
    policy, host = policy_for(Adaptive(horizon))
    with pytest.raises(ValueError, match="Selected execution horizon"):
        policy.step(observation())
    assert policy.executed_count == 0
    assert not policy._queue
    assert host.calls == 1


def test_reset_and_instruction_change_clear_selected_prefix():
    method = Adaptive(16)
    policy, _ = policy_for(method)
    policy.step(observation())
    assert policy.selected_horizon == 16
    policy.reset(EpisodeContext("suite", 0, 1))
    assert policy.selected_horizon is None and method.seen == []
    policy.step(observation())
    assert policy.step(replace(observation(), instruction="new")).queried
    assert len(method.seen) == 1


def test_interrupt_discards_real_remaining_prefix():
    method = Adaptive(16)
    policy, _ = policy_for(method)
    method.after_action = lambda action, remaining: remaining == 14
    decisions = [policy.step(observation()) for _ in range(3)]
    assert [d.queried for d in decisions] == [True, False, True]
    assert decisions[1].diagnostics["replan_requested"]


def test_default_and_detach_are_native_eight():
    method = Adaptive(3)
    policy, host = policy_for(method)
    policy.step(observation())
    host.detach_method()
    policy.reset(EpisodeContext("suite", 0, 1))
    decisions = [policy.step(observation()) for _ in range(9)]
    assert [i for i, d in enumerate(decisions) if d.queried] == [0, 8]
    assert all(d.diagnostics["selected_execution_horizon"] == 8 for d in decisions)
    assert InferenceMethod().select_execution_horizon(np.zeros((16, 7)), 8) == 8


def test_adaptive_selection_cannot_hide_short_native_output():
    policy, _ = policy_for(Adaptive(2), length=4)
    with pytest.raises(ValueError, match="enough finite"):
        policy.step(observation())


def test_closed_host_refuses_selection():
    host = Host()
    host.close()
    with pytest.raises(RuntimeError, match="closed"):
        host.select_execution_horizon(np.zeros((16, 7)), 8)
