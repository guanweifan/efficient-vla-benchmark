import numpy as np
import pytest

from ..config import AACConfig


@pytest.mark.gpu
def test_native_disabled_reset_candidate_cost_and_detach(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import AAC

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    host.attach_method(AAC(host, AACConfig(enabled=False)))
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
    host.detach_method()
    calls = []
    handle = host.model.vlm.register_forward_hook(lambda *args: calls.append(1))
    request.addfinalizer(handle.remove)
    for k in (5, 20):
        host.attach_method(AAC(host, AACConfig(num_samples=k)))
        values = []
        for _ in range(2):
            host.reset(context)
            calls.clear()
            values.append(host.predict_chunk(frames[0]))
            assert len(calls) == 1
            assert values[-1].shape == (16, 7) and np.isfinite(values[-1]).all()
            assert host.query_diagnostics["action_candidates"] == k
            assert host.query_diagnostics["action_head_batch"] == 2 * k
            assert host.query_diagnostics["denoiser_calls"] == 10
            assert 2 <= host.select_execution_horizon(values[-1], 8) <= 16
        np.testing.assert_array_equal(*values)
        policy = ChunkPolicy(host, 8)
        policy.reset(context)
        decisions = [policy.step(frame) for frame in frames[:32]]
        assert policy.query_count == sum(d.queried for d in decisions)
        assert all(2 <= d.diagnostics["selected_execution_horizon"] <= 16 for d in decisions)
        host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
