"""CogACT binding: native action codec and an independent physical-time committee."""

import numpy as np
import pytest

from efficient_vla_benchmark.core.config import ExperimentConfig

from ..codec import decode, encode
from ..config import VoteConfig
from ..implementations.cogact.api import build, validate_experiment


def test_cogact_history_bound_and_paper_cadence():
    validate_experiment(ExperimentConfig("cogact", execution_horizon=1), VoteConfig(max_history=15))
    with pytest.raises(ValueError, match="fifteen"):
        validate_experiment(
            ExperimentConfig("cogact", execution_horizon=1), VoteConfig(max_history=16)
        )
    with pytest.raises(ValueError, match="horizon=1"):
        validate_experiment(ExperimentConfig("cogact", execution_horizon=8), VoteConfig())


def test_shared_codec_and_sixteen_step_alignment():
    method = build(None, VoteConfig(max_history=15), None)
    chunk = np.zeros((16, 7))
    chunk[:, 0] = np.arange(1, 17)
    chunk[:, -1] = -1
    method.on_plan(chunk, 0)
    np.testing.assert_array_equal(method.transform_action(chunk[15], 15), chunk[15])
    assert method.candidate_count == 1
    np.testing.assert_array_equal(np.stack([decode(row) for row in encode(chunk)]), chunk)
    method.reset(None)
    assert not method.predictions


def reference(candidates):
    newest = candidates[-1]
    yes, no = [], []
    for action in candidates:
        norm = np.linalg.norm(action) * np.linalg.norm(newest)
        similarity = (
            float(action @ newest) / norm if norm else float(np.array_equal(action, newest))
        )
        (yes if similarity > 0.5 else no).append(action)
    return np.mean(yes if len(yes) > len(no) else no, axis=0)


@pytest.mark.gpu
def test_real_cogact_voting_noop_reset_and_diffusion_rng(model_path, trace_path, request):
    import torch
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = CogACTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path, require_control_steps=True)
    observations = observations[:12]
    backend.reset(context)
    chunks = [backend.predict_chunk(obs) for obs in observations]
    native_rng = torch.cuda.get_rng_state().clone()
    assert all(chunk.shape == (16, 7) for chunk in chunks)

    backend.attach_method(build(backend, VoteConfig(enabled=False), None))
    policy = ChunkPolicy(backend, 1)
    policy.reset(context)
    for obs, chunk in zip(observations, chunks):
        np.testing.assert_array_equal(policy.step(obs).action, chunk[0])
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    backend.detach_method()

    backend.attach_method(build(backend, VoteConfig(), None))
    policy.reset(context)
    actual = []
    for step, obs in enumerate(observations):
        candidates = [
            encode(chunks[start])[step - start] for start in range(max(0, step - 3), step + 1)
        ]
        decision = policy.step(obs)
        np.testing.assert_allclose(
            decision.action, decode(reference(candidates)), rtol=0, atol=1e-12
        )
        assert decision.queried and backend.method.candidate_count == min(step + 1, 4)
        actual.append(decision.action.copy())
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    assert any(not np.array_equal(action, chunk[0]) for action, chunk in zip(actual, chunks))
    policy.reset(context)
    for obs, expected in zip(observations, actual):
        np.testing.assert_array_equal(policy.step(obs).action, expected)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), chunks[0])
