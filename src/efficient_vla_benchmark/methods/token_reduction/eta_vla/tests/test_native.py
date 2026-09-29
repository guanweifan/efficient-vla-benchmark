"""Native models: dense parity, active shortening, CFG/RNG, reset and detach."""

import importlib

import numpy as np
import pytest
import torch

from ..config import ILSAConfig


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["openvla_oft", "cogact"])
def test_real_model_dense_active_reset_and_detach(backend, model_path, trace_path, request):
    module = pytest.importorskip(f"efficient_vla_backend_{backend}.backend")
    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend_class = module.CogACTBackend if backend == "cogact" else module.OpenVLAOFTBackend
    host = backend_class(model_path)
    request.addfinalizer(host.close)
    build = importlib.import_module(
        f"efficient_vla_benchmark.methods.token_reduction.eta_vla.implementations.{backend}.api"
    ).build
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (ILSAConfig(enabled=False), ILSAConfig(keep_ratio=1), ILSAConfig(keep_ratio=0.5)):
        method = build(host, config, None)
        host.attach_method(method)
        repeats = []
        for _ in range(2):
            host.reset(context)
            result = host.predict_chunk(frames[0])
            repeats.append(result)
            assert result.shape == native.shape and np.isfinite(result).all()
            assert torch.equal(torch.cuda.get_rng_state(), native_rng)
            if not config.enabled or config.keep_ratio == 1:
                np.testing.assert_array_equal(result, native)
            else:
                d = host.query_diagnostics
                assert d["ilsa"]["text_anchors"] > 0
                assert d["ilsa"]["semantic_tokens"] > 0 and d["ilsa"]["recycled_tokens"] > 0
                expected = 128 if backend == "cogact" else 256
                assert d["visual_tokens_final"] == expected
                sizes = d["layer_sequence_lengths"]
                assert sizes[3] - sizes[4] == expected
            assert method.pending is None
        np.testing.assert_array_equal(repeats[0], repeats[1])
        host.detach_method()
        method.close()
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
