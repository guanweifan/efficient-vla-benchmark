"""Real native models: prefix accounting, H-control refresh, RNG and detach."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from ..config import TIESConfig


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["openvla_oft", "cogact"])
def test_native_prefix_refresh_active_reset_and_detach(backend, model_path, trace_path, request):
    module = pytest.importorskip(f"efficient_vla_backend_{backend}.backend")
    from efficient_vla_benchmark.evaluation.traces import load_trace

    binding = importlib.import_module(
        f"efficient_vla_benchmark.methods.token_reduction.ties.implementations.{backend}.method"
    ).TIES
    cls = module.CogACTBackend if backend == "cogact" else module.OpenVLAOFTBackend
    host = cls(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    # A test-only equation fixture, never a calibration/evaluation artifact.
    manifest = dict(
        quantiles=[-1.0, 0.0, 1.0],
        statistics_sha256="test-only",
        samples_sha256="test-only",
        checkpoint_sha256="test-only",
    )
    base = TIESConfig(ranking_topk=192 if backend == "openvla_oft" else 30)
    for config in (replace(base, enabled=False), replace(base, keep_ratio=1), base):
        method = binding(host, config, manifest)
        host.attach_method(method)
        outputs = []
        for _ in range(2):
            host.reset(context)
            result = host.predict_chunk(frames[0])
            outputs.append(result)
            assert np.isfinite(result).all() and result.shape == native.shape
            assert torch.equal(torch.cuda.get_rng_state(), native_rng)
            if not method.active:
                np.testing.assert_array_equal(native, result)
            else:
                d = host.query_diagnostics
                assert d["ties"]["extra_prefix_layers"] == 16
                assert d["ties"]["tau"] is not None
                assert len(method.rankings) == 16
                expected = 128 if backend == "cogact" else 256
                assert d["visual_tokens_final"] == expected
                assert d["layer_sequence_lengths"][1] - d["layer_sequence_lengths"][2] == expected
                # Refresh is based on executed controls, not model query count.
                for step in range(1, 25):
                    method.after_action(np.zeros(7), 0)
                    if step in (8, 16, 24):
                        host.predict_chunk(frames[step % len(frames)])
                        expected_prefix = 16 if step == 24 else 0
                        assert (
                            host.query_diagnostics["ties"]["extra_prefix_layers"] == expected_prefix
                        )
                assert method.last_refresh == 24 and not method.probing
        np.testing.assert_array_equal(outputs[0], outputs[1])
        host.detach_method()
        method.close()
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
