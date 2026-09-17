import numpy as np
import pytest


@pytest.mark.gpu
def test_native_chunk_gate_and_noop_parity(model_path, trace_path):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..config import SPVLAConfig
    from ..method import SPVLAScheduling

    context, observations, _ = load_trace(trace_path, require_control_steps=True)
    backend = OpenVLAOFTBackend(model_path)
    try:
        policy = ChunkPolicy(backend, 8)
        policy.reset(context)
        expected = [policy.step(o).action for o in observations[:24]]
        backend.attach_method(SPVLAScheduling(SPVLAConfig.parse({"enabled": False})))
        policy.reset(context)
        for obs, reference in zip(observations[:24], expected):
            np.testing.assert_array_equal(policy.step(obs).action, reference)
        backend.detach_method()
        backend.attach_method(SPVLAScheduling(SPVLAConfig.parse({})))
        policy.reset(context)
        decisions = [policy.step(o) for o in observations[:64]]
        assert all(np.isfinite(d.action).all() for d in decisions)
        assert policy.query_count == sum(d.queried for d in decisions)
        assert backend.method.replacements > 0
        assert policy.query_count <= 8
        policy.reset(context)
        np.testing.assert_array_equal(policy.step(observations[0]).action, expected[0])
    finally:
        backend.close()
