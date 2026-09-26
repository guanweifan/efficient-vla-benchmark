import importlib

import numpy as np
import pytest
import torch

from efficient_vla_benchmark.core.backend import RuntimeResources

from ..config import EcoVLAConfig


@pytest.mark.gpu
@pytest.mark.parametrize("backend_id", ["openvla_oft", "cogact"])
def test_real_eco_dense_sparse_reset_rng_and_restore(
    backend_id, model_path, trace_path, artifact_path, request
):
    plugin = pytest.importorskip(f"efficient_vla_backend_{backend_id}.backend")
    cls = getattr(plugin, "OpenVLAOFTBackend" if backend_id == "openvla_oft" else "CogACTBackend")
    api = importlib.import_module(
        f"efficient_vla_benchmark.methods.model_routing.eco_vla.implementations.{backend_id}.api"
    )
    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = cls(model_path)
    request.addfinalizer(backend.close)
    resources = RuntimeResources(model_path, artifact_path=artifact_path)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    backend.predict_chunk(observations[0])
    backend.predict_chunk(observations[0])
    rng = torch.cuda.get_rng_state().clone()
    for config in (EcoVLAConfig(enabled=False), EcoVLAConfig(keep_ratio=1)):
        backend.attach_method(api.build(backend, config, resources))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    method = api.build(backend, EcoVLAConfig(trigger_quantile=0), resources)
    backend.attach_method(method)
    backend.reset(context)
    actions, modes = [], []
    for _ in range(3):
        actions.append(backend.predict_chunk(observations[0]))
        backend.runtime.synchronize()
        modes.append(backend.query_diagnostics["eco_dense_refresh"])
    assert modes == [True, True, False]
    np.testing.assert_allclose(actions[0], native, atol=1e-6, rtol=0)
    assert np.isfinite(actions[2]).all() and actions[2].shape == native.shape
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert backend.query_diagnostics["eco_attention_heads"] == 24
    assert backend.query_diagnostics["eco_mlp_channels"] == 8256
    assert method.pending is None and not method.inflight
    backend.reset(context)
    for expected in actions:
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), expected)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
