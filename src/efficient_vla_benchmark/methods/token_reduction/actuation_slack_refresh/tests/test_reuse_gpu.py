"""Real checkpoint parity, sparse execution, reset, detach and RNG contracts."""

import importlib

import numpy as np
import pytest
import torch

from ..config import ASRConfig


@pytest.mark.gpu
@pytest.mark.parametrize("backend_id", ["openvla_oft", "cogact"])
def test_clean_reuse_real_model(backend_id, model_path, trace_path, request):
    module = pytest.importorskip(f"efficient_vla_backend_{backend_id}.backend")
    cls = getattr(module, "OpenVLAOFTBackend" if backend_id == "openvla_oft" else "CogACTBackend")
    method_cls = importlib.import_module(
        f"efficient_vla_benchmark.methods.token_reduction.actuation_slack_refresh"
        f".implementations.{backend_id}.reuse"
    ).ActuationSlackReuse
    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = cls(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    pair = [observations[0], observations[min(8, len(observations) - 1)]]
    backend.reset(context)
    native = backend.predict_chunk(pair[0])
    rng_first = torch.cuda.get_rng_state().clone()
    backend.predict_chunk(pair[1])
    rng_second = torch.cuda.get_rng_state().clone()
    for config in (ASRConfig(enabled=False), ASRConfig(keep_ratio=1)):
        backend.attach_method(method_cls(backend, config))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(pair[0]), native)
        assert backend.auxiliary_runner is None
        backend.detach_method()
    backend.attach_method(method_cls(backend, ASRConfig()))
    backend.reset(context)
    first = backend.predict_chunk(pair[0])
    backend.runtime.synchronize()
    np.testing.assert_allclose(first, native, atol=1e-6, rtol=0)
    assert torch.equal(torch.cuda.get_rng_state(), rng_first)
    gate = backend.method.state.gate.clone()
    kv = {layer: value.clone() for layer, value in backend.method.store.keys.items()}
    second = backend.predict_chunk(pair[1])
    backend.runtime.synchronize()
    assert second.shape == native.shape and np.isfinite(second).all()
    assert torch.equal(torch.cuda.get_rng_state(), rng_second)
    diagnostic = backend.query_diagnostics
    assert diagnostic["visual_query_tokens_final"] == (282 if backend_id == "openvla_oft" else 26)
    assert diagnostic["visual_kv_tokens"] == (512 if backend_id == "openvla_oft" else 256)
    assert backend.method.state.refreshes == 2
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(pair[0]), first)
    backend.runtime.synchronize()
    torch.testing.assert_close(backend.method.state.gate, gate, rtol=0, atol=0)
    for layer, value in kv.items():
        torch.testing.assert_close(backend.method.store.keys[layer], value, rtol=0, atol=0)
    np.testing.assert_array_equal(backend.predict_chunk(pair[1]), second)
    backend.detach_method()
    assert backend.auxiliary_runner is None
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(pair[0]), native)
