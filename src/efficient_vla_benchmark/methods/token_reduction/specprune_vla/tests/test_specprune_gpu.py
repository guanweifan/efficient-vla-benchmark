import numpy as np
import pytest


@pytest.mark.gpu
def test_specprune_real_model_noop_full_mechanism_and_reset(model_path, trace_path):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace
    from efficient_vla_benchmark.methods.token_reduction.specprune_vla.config import SpecPruneConfig

    from ..implementations.openvla_oft.method import (
        SpecPrune,
    )

    context, samples, _ = load_trace(trace_path, require_control_steps=True)
    backend = OpenVLAOFTBackend(model_path)
    backend.reset(context)
    reference = backend.predict_chunk(samples[0])
    no_op = SpecPrune(
        backend,
        SpecPruneConfig.parse(
            {
                "static_pruning": False,
                "dynamic_pruning": False,
                "action_controller": False,
            }
        ),
    )
    backend.attach_method(no_op)
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), reference, atol=1e-6, rtol=0)
    backend.detach_method()

    method = SpecPrune(backend, SpecPruneConfig.parse({}))
    backend.attach_method(method)
    policy = ChunkPolicy(backend, 8)
    results = []
    for _ in range(2):
        policy.reset(context)
        assert method.confidence == {} and len(method.frames) == 0
        commands, diagnostics = [], []
        for sample in samples[:32]:
            decision = policy.step(sample)
            assert np.isfinite(decision.action).all()
            commands.append(decision.action)
            if decision.queried:
                diagnostics.append(dict(backend.query_diagnostics))
        assert len(diagnostics) >= 4
        assert all(0 < d["final_visual_tokens"] < 512 for d in diagnostics)
        assert all(len(d["tokens_per_layer"]) == 32 for d in diagnostics)
        assert any(d["global_reuse_tokens"] > 0 for d in diagnostics)
        assert any(d["dynamic_pruning_layers"] for d in diagnostics)
        assert any(d["importance_updates"] for d in diagnostics)
        assert method.control_steps == min(32, len(samples))
        results.append(np.stack(commands))
    np.testing.assert_allclose(results[0], results[1], atol=1e-6, rtol=0)
    backend.close()
