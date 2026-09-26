"""H8 native chunks with an independently aligned two-vote reference."""

import numpy as np
import pytest

from efficient_vla_benchmark.core.config import ExperimentConfig

from ..codec import decode, encode
from ..config import VoteConfig
from ..implementations.cogact.api import build, validate_experiment


def test_cogact_history_bound_and_h8_cadence():
    validate_experiment(ExperimentConfig("cogact", execution_horizon=8), VoteConfig())
    with pytest.raises(ValueError, match="one overlapping"):
        validate_experiment(
            ExperimentConfig("cogact", execution_horizon=8), VoteConfig(history_chunks=2)
        )
    with pytest.raises(ValueError, match="horizon=8"):
        validate_experiment(ExperimentConfig("cogact", execution_horizon=1), VoteConfig())


def reference(candidates, threshold=0.5):
    newest = candidates[-1]
    yes, no = [], []
    for action in candidates:
        norm = np.linalg.norm(action) * np.linalg.norm(newest)
        similarity = (
            float(action @ newest) / norm if norm else float(np.array_equal(action, newest))
        )
        (yes if similarity > threshold else no).append(action)
    return np.mean(yes if len(yes) > len(no) else no, axis=0)


def test_complete_h8_chunks_expiry_and_original_prediction_storage():
    method = build(None, VoteConfig(), None)
    chunks = []
    for start in (0, 8, 16, 24):
        chunk = np.zeros((16, 7))
        chunk[:, 0] = (np.arange(16) + start + 1) * (-1 if start == 8 else 1)
        chunk[:, -1] = -1
        chunks.append(chunk.copy())
        method.on_plan(chunk, start)
        chunk[:] = 999  # No alias to controller-owned data.
        for step in range(start, start + 8):
            candidates = [
                encode(c)[step - s]
                for s, c in zip(range(0, start + 1, 8), chunks)
                if s <= step < s + 16
            ]
            np.testing.assert_array_equal(
                method.transform_action(chunks[-1][step - start], step),
                decode(reference(candidates)),
            )
            assert method.candidate_count == (1 if start == 0 else 2)
            assert len(method.predictions) <= 2
    with pytest.raises(ValueError, match="strictly increasing"):
        method.on_plan(chunks[-1], 24)
    method.reset(None)
    assert not method.predictions and method.candidate_count == 0


@pytest.mark.gpu
def test_real_cogact_voting_noop_reset_and_diffusion_rng(model_path, trace_path, request):
    import torch
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = CogACTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path, require_control_steps=True)
    observations = observations[:24]
    assert len(observations) == 24
    backend.reset(context)
    chunks = {step: backend.predict_chunk(observations[step]) for step in (0, 8, 16)}
    native_rng = torch.cuda.get_rng_state().clone()
    assert all(chunk.shape == (16, 7) for chunk in chunks.values())

    backend.attach_method(build(backend, VoteConfig(enabled=False), None))
    policy = ChunkPolicy(backend, 8)
    policy.reset(context)
    for step, obs in enumerate(observations):
        decision = policy.step(obs)
        np.testing.assert_array_equal(decision.action, chunks[step // 8 * 8][step % 8])
        assert decision.queried == (step % 8 == 0)
    assert policy.query_count == 3
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    backend.detach_method()

    backend.attach_method(build(backend, VoteConfig(), None))
    policy.reset(context)
    actual = []
    for step, obs in enumerate(observations):
        candidates = [
            encode(chunk)[step - start]
            for start, chunk in chunks.items()
            if start <= step < start + 16
        ]
        decision = policy.step(obs)
        np.testing.assert_allclose(
            decision.action, decode(reference(candidates)), rtol=0, atol=1e-12
        )
        assert decision.queried == (step % 8 == 0)
        assert backend.method.candidate_count == (1 if step < 8 else 2)
        actual.append(decision.action.copy())
    assert policy.query_count == 3
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    assert any(
        not np.array_equal(action, chunks[step // 8 * 8][step % 8])
        for step, action in enumerate(actual)
    )
    policy.reset(context)
    for obs, expected in zip(observations, actual):
        np.testing.assert_array_equal(policy.step(obs).action, expected)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), chunks[0])
