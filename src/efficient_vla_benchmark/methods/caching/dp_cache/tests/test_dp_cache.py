import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import DPCacheConfig
from ..implementations.cogact.method import DPCache


def fixture():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model.models import DiT

    torch.manual_seed(3)
    model = DiT(hidden_size=24, depth=2, num_heads=3, future_action_window_size=3).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
    return model, SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=model), query_diagnostics={}
    )


def test_output_broadcast_cfg_reset_rng_and_detach():
    model, host = fixture()
    reference = copy.deepcopy(model)
    method = DPCache(host, DPCacheConfig())
    with torch.inference_mode():
        x, z = torch.randn(2, 4, 7), torch.randn(2, 1, 4096)
        rng = torch.get_rng_state().clone()
        for _ in range(2):
            method.before_query({})
            for i in range(10):
                t = torch.full((2,), 90 - 10 * i)
                if i not in (3, 5, 7):
                    expected = reference.forward_with_cfg(x, t, z, 1.5)
                actual = model.forward_with_cfg(x, t, z, 1.5)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            method.after_query(actual)
            assert host.query_diagnostics["denoiser_evaluations"] == 7
            assert host.query_diagnostics["denoiser_output_reuses"] == 3
            assert method.saved is None
        assert torch.equal(rng, torch.get_rng_state())
        method.close()
        method.close()
        assert "forward" not in model.__dict__
        torch.testing.assert_close(model(x, t, z), reference(x, t, z), rtol=0, atol=0)


def test_noop_error_and_config():
    model, host = fixture()
    for config in (DPCacheConfig(enabled=False), DPCacheConfig(cache_interval=1)):
        m = DPCache(host, config)
        assert not m.patches.saved
        m.close()
    m = DPCache(host, DPCacheConfig())
    with pytest.raises(RuntimeError, match="nonnested"):
        model(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
    m.before_query({})
    with pytest.raises(RuntimeError, match="inference-only"):
        model(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
    assert not m.query_open
    m.close()
    for bad in (
        {"cache_interval": True},
        {"cache_interval": 0},
        {"end_step": -1},
        {"enabled": 1},
        {"foo": 0},
    ):
        with pytest.raises(ValueError):
            DPCacheConfig.parse(bad)


@pytest.mark.gpu
def test_real_native_noop_active_counts_reset(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    for cfg in (DPCacheConfig(enabled=False), DPCacheConfig(cache_interval=1)):
        host.attach_method(DPCache(host, cfg))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        host.detach_method()
    host.attach_method(DPCache(host, DPCacheConfig()))
    results = []
    for _ in range(2):
        host.reset(context)
        results.append(host.predict_chunk(frames[0]))
        assert np.isfinite(results[-1]).all()
        assert host.query_diagnostics["denoiser_evaluations"] == 7
        assert host.query_diagnostics["denoiser_calls"] == 10
    np.testing.assert_array_equal(*results)
    assert not np.array_equal(native, results[0])
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
