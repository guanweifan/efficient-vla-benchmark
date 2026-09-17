import numpy as np
import pytest


@pytest.mark.gpu
def test_real_prediction_alignment_and_disabled_parity(model_path, trace_path):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..config import VoteConfig
    from ..implementations.openvla_oft.api import decode, encode
    from ..method import VoteEnsemble, ensemble_vote

    context, observations, _ = load_trace(trace_path, require_control_steps=True)
    backend = OpenVLAOFTBackend(model_path)
    try:
        backend.reset(context)
        chunks = [backend.predict_chunk(o) for o in observations[:12]]
        backend.attach_method(VoteEnsemble(VoteConfig(enabled=False), encode, decode))
        policy = ChunkPolicy(backend, 1)
        policy.reset(context)
        for obs, chunk in zip(observations, chunks):
            np.testing.assert_array_equal(policy.step(obs).action, chunk[0])
        backend.detach_method()
        backend.attach_method(VoteEnsemble(VoteConfig(), encode, decode))
        policy.reset(context)
        changed = 0
        for t, obs in enumerate(observations[:12]):
            candidates = [encode(chunks[s])[t - s] for s in range(max(0, t - 3), t + 1)]
            expected, _ = ensemble_vote(candidates, 0.5)
            decision = policy.step(obs)
            np.testing.assert_array_equal(decision.action, decode(expected))
            assert decision.queried
            changed += int(not np.array_equal(decision.action, chunks[t][0]))
        assert changed > 0 and policy.query_count == 12
        policy.reset(context)
        assert not backend.method.predictions
    finally:
        backend.close()
