import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import C3acheConfig
from ..implementations.cogact.method import C3ache


def test_native_suffix_reference_current_condition_reset_and_detach():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model.models import DiT

    torch.manual_seed(22)
    model = DiT(hidden_size=24, depth=3, num_heads=3, future_action_window_size=3).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
    reference = copy.deepcopy(model)
    host = SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=model), query_diagnostics={}
    )
    method = C3ache(host, C3acheConfig(cache_steps=2, refresh_interval=2))
    cache = {}
    with torch.inference_mode():
        x = torch.randn(2, 4, 7)
        for chunk in range(3):
            method.before_query({})
            for step in range(4):
                z = torch.randn(2, 1, 4096)
                t = torch.full((2,), 90 - step * 10)
                embedded = reference.x_embedder(x)
                cond = reference.t_embedder(t)[:, None] + reference.z_embedder(z, False)
                h = torch.cat((cond, embedded), 1) + reference.positional_embedding
                h = reference.blocks[0](h)
                anchor = h
                if chunk == 1 and step < 2:
                    h = h + cache[step]
                else:
                    for block in reference.blocks[1:]:
                        h = block(h)
                    if step < 2:
                        cache[step] = h - anchor
                expected = reference.final_layer(h)[:, 1:]
                actual = model(x, t, z)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            method.after_query(actual)
            assert host.query_diagnostics["residual_reuse_steps"] == (2 if chunk == 1 else 0)
        # A reused step must still depend on current cognition, not just old residual.
        method.before_query({})
        z = torch.randn(2, 1, 4096)
        a = model(x, t, z)
        method.step = 0
        b = model(x, t, z + 2)
        assert not torch.equal(a, b)
        method.reset(None)
        assert not method.cache and method.chunk == 0
        method.close()
        method.close()
        torch.testing.assert_close(model(x, t, z), reference(x, t, z), rtol=0, atol=0)
        assert all("forward" not in block.__dict__ for block in model.blocks)


@pytest.mark.gpu
def test_real_cogact_conditioned_cache(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    host.attach_method(C3ache(host, C3acheConfig(refresh_interval=1)))
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
    host.detach_method()
    host.attach_method(C3ache(host, C3acheConfig()))
    outputs = []
    for _ in range(2):
        host.reset(context)
        first = host.predict_chunk(frames[0])
        np.testing.assert_array_equal(native, first)
        second = host.predict_chunk(frames[8])
        assert np.isfinite(second).all()
        assert host.query_diagnostics["residual_reuse_steps"] == 7
        assert host.query_diagnostics["cache_entries"] == 7
        outputs.append(second)
    np.testing.assert_array_equal(*outputs)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
