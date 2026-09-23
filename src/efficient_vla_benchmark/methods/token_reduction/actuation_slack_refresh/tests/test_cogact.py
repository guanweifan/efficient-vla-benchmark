import numpy as np
import pytest
import torch

from ..config import ASRConfig

pytest.importorskip("efficient_vla_backend_cogact")


@pytest.mark.gpu
def test_real_cogact_refresh_preserves_native_noise_and_clean_gate(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import ActuationSlackRefresh

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, observations, _ = load_trace(trace_path)
    second_observation = observations[min(8, len(observations) - 1)]
    host.reset(context)
    native = host.predict_chunk(observations[0])
    native_first_rng = torch.cuda.get_rng_state().clone()
    host.predict_chunk(second_observation)
    native_second_rng = torch.cuda.get_rng_state().clone()
    for config in (ASRConfig(enabled=False), ASRConfig(keep_ratio=1)):
        host.attach_method(ActuationSlackRefresh(host, config))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
        assert host.auxiliary_runner is None
        host.detach_method()
    host.attach_method(ActuationSlackRefresh(host, ASRConfig()))
    host.reset(context)
    first = host.predict_chunk(observations[0])
    host.runtime.synchronize()
    np.testing.assert_array_equal(first, native)
    assert torch.equal(torch.cuda.get_rng_state(), native_first_rng)
    assert host.query_diagnostics["visual_tokens_final"] == 256
    assert host.query_diagnostics["auxiliary_visual_tokens_final"] == 256
    assert host.method.state.refreshes == 1
    gate = host.method.state.gate.clone()
    second = host.predict_chunk(second_observation)
    host.runtime.synchronize()
    assert second.shape == (16, 7) and np.isfinite(second).all()
    assert host.query_diagnostics["visual_tokens_final"] == 26
    assert host.method.state.refreshes == 2
    assert torch.equal(torch.cuda.get_rng_state(), native_second_rng)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), first)
    host.runtime.synchronize()
    torch.testing.assert_close(host.method.state.gate, gate, rtol=0, atol=0)
    np.testing.assert_array_equal(host.predict_chunk(second_observation), second)
    host.detach_method()
    assert host.auxiliary_runner is None
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
