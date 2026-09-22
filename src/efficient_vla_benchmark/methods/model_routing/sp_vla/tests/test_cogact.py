"""Native no-op, active pruning, scheduler cadence and RNG checks."""

import numpy as np
import pytest

from ..config import SPVLAConfig
from ..implementations.cogact.api import build
from ..method import SPVLAScheduling


def test_disabled_and_scheduling_only_do_not_import_backend():
    assert isinstance(build(None, SPVLAConfig(enabled=False), None), SPVLAScheduling)
    method = build(None, SPVLAConfig(), None)
    action = np.array([0.3, 0.1, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(8):
        method.after_action(action, 0)
    assert method.before_plan(None, 8) is not None
    assert method.replacements == 1
    assert method.origins.count(True) == 6
    method.reset(None)
    assert not method.history


@pytest.mark.gpu
def test_native_cogact_pruning_reset_and_scheduler(model_path, trace_path, request):
    import torch
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import SPVLA

    backend = CogACTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path, require_control_steps=True)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    native_rng = torch.cuda.get_rng_state().clone()
    backend.attach_method(SPVLA(backend, SPVLAConfig(token_pruning=True)))
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    assert backend.query_diagnostics["visual_tokens_final"] == 256
    backend.reset(context)
    backend.method.after_action(np.array([0, 0, 1, 0, 0, 0, -1.0]), 0)
    pruned = backend.predict_chunk(samples[0])
    assert pruned.shape == (16, 7) and np.isfinite(pruned).all()
    assert backend.query_diagnostics["visual_tokens_final"] < 256
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    # Native reset recovers cold dense behavior and clears scheduler history.
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    policy = ChunkPolicy(backend, 8)
    policy.reset(context)
    decisions = [policy.step(obs) for obs in samples[:80]]
    assert sum(d.queried for d in decisions) > 0
    assert backend.method.replacements > 0
    assert sum(d.queried for d in decisions) == (80 - backend.method.replacements + 7) // 8
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
