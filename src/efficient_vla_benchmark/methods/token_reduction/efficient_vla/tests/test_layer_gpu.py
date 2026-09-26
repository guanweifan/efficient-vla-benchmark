import importlib

import numpy as np
import pytest
import torch

from efficient_vla_benchmark.core.backend import RuntimeResources

from ..layer_config import LayerPruningConfig


@pytest.mark.gpu
@pytest.mark.parametrize("backend_id", ["openvla_oft", "cogact"])
def test_real_compaction_noop_active_reset_and_detach(
    backend_id, model_path, trace_path, artifact_path, request
):
    plugin = pytest.importorskip(f"efficient_vla_backend_{backend_id}.backend")
    backend_cls = getattr(
        plugin, "OpenVLAOFTBackend" if backend_id == "openvla_oft" else "CogACTBackend"
    )
    api = importlib.import_module(
        "efficient_vla_benchmark.methods.token_reduction.efficient_vla.implementations."
        f"{backend_id}.layers"
    )
    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = backend_cls(model_path)
    request.addfinalizer(backend.close)
    resources = RuntimeResources(model_path, artifact_path=artifact_path)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    rng = torch.cuda.get_rng_state().clone()
    for config in (LayerPruningConfig(enabled=False), LayerPruningConfig(prune_layers=0)):
        backend.attach_method(api.build(backend, config, resources))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    method = api.build(backend, LayerPruningConfig(), resources)
    backend.attach_method(method)
    assert len(method.decoder.layers) == 28
    assert all(next(method.original[i].parameters()).device.type == "cpu" for i in method.removed)
    backend.reset(context)
    pruned = backend.predict_chunk(observations[0])
    assert pruned.shape == native.shape and np.isfinite(pruned).all()
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), pruned)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
