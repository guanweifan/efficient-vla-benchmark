import numpy as np
import pytest
import torch

from ..config import ASRConfig


@pytest.mark.gpu
def test_real_oft_clean_gate_refresh_reset_and_restore(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import ActuationSlackRefresh

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (ASRConfig(enabled=False), ASRConfig(keep_ratio=1)):
        backend.attach_method(ActuationSlackRefresh(backend, config))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        assert backend.auxiliary_runner is None
        backend.detach_method()
    backend.attach_method(ActuationSlackRefresh(backend, ASRConfig()))
    backend.reset(context)
    first = backend.predict_chunk(observations[0])
    backend.runtime.synchronize()
    np.testing.assert_array_equal(first, native)
    assert backend.method.state.refreshes == 1
    assert backend.query_diagnostics["visual_tokens_final"] == 512
    assert backend.query_diagnostics["auxiliary_visual_tokens_final"] == 512
    assert backend.query_diagnostics["auxiliary_forward_ms"] > 0
    gate = backend.method.state.gate.clone()
    second = backend.predict_chunk(observations[min(8, len(observations) - 1)])
    backend.runtime.synchronize()
    assert second.shape == (8, 7) and np.isfinite(second).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 282
    assert backend.method.state.refreshes == 2 and not backend.method.state.refreshing
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), first)
    backend.runtime.synchronize()
    torch.testing.assert_close(backend.method.state.gate, gate, rtol=0, atol=0)
    np.testing.assert_array_equal(
        backend.predict_chunk(observations[min(8, len(observations) - 1)]), second
    )
    # Detach must join the still-pending refresh before removing any hooks.
    backend.detach_method()
    assert backend.auxiliary_runner is None
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
