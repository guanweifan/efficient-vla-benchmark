"""Independent recurrence, native CFG, no-op, query reset and real-model checks."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..cache_config import DiffusionCacheConfig
from ..implementations.cogact.method import EfficientVLADiffusionCache


def native_dit():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model.models import DiT

    torch.manual_seed(12)
    model = DiT(hidden_size=24, depth=2, num_heads=3, future_action_window_size=3).eval()
    # Native initialization zeroes the last projection. Nonzero output is essential:
    # a zero-output fixture would falsely certify every caching implementation.
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
    return model


def host_for(model):
    return SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=model), query_diagnostics={}
    )


def reference_cfg(model, x, t, z, cache, refresh, scale):
    half = x[: len(x) // 2]
    x = model.x_embedder(torch.cat([half, half], 0))
    condition = model.t_embedder(t)[:, None] + model.z_embedder(z, False)
    x = torch.cat((condition, x), 1) + model.positional_embedding
    for index, block in enumerate(model.blocks):
        if refresh:
            attention = block.attn(block.norm1(x))
            mlp = block.mlp(block.norm2(x + attention))
            cache[index] = attention, mlp
        attention, mlp = cache[index]
        # Inputs and residuals remain live even when BOTH feature outputs are cached.
        x = (x + attention) + mlp
    output = model.final_layer(x)[:, 1:]
    cond, uncond = output.chunk(2)
    guided = uncond + scale * (cond - uncond)
    return torch.cat((guided, guided), 0)


def test_periodic_features_match_independent_recurrence_and_cfg():
    model = native_dit()
    reference = copy.deepcopy(model)
    host = host_for(model)
    method = EfficientVLADiffusionCache(host, DiffusionCacheConfig(cache_interval=3))
    with pytest.raises(ValueError, match="Only one"):
        EfficientVLADiffusionCache(host, DiffusionCacheConfig())
    generator = torch.Generator().manual_seed(47)
    inputs = [
        (
            torch.randn(2, 4, 7, generator=generator),
            torch.full((2,), 90 - index * 10),
            torch.randn(2, 1, 4096, generator=generator),
        )
        for index in range(10)
    ]
    sequences = []
    try:
        with torch.inference_mode():
            rng = torch.get_rng_state().clone()
            for _ in range(2):
                method.before_query({})
                cache, outputs = {}, []
                for index, (x, t, z) in enumerate(inputs):
                    expected = reference_cfg(reference, x, t, z, cache, index % 3 == 0, 1.5)
                    actual = model.forward_with_cfg(x, t, z, 1.5)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    outputs.append(actual)
                # Conditional/unconditional feature rows must not be collapsed.
                assert not torch.equal(method.cache[0, "attn"][1][0], method.cache[0, "attn"][1][1])
                method.after_query(outputs[-1])
                assert not method.cache and not method.query_open
                assert host.query_diagnostics["denoiser_calls"] == 10
                assert host.query_diagnostics["feature_refresh_steps"] == 4
                assert host.query_diagnostics["attention_evaluations"] == 8
                assert host.query_diagnostics["mlp_evaluations"] == 8
                assert host.query_diagnostics["attention_reuses"] == 12
                sequences.append(torch.stack(outputs))
            torch.testing.assert_close(sequences[0], sequences[1], rtol=0, atol=0)
            assert torch.equal(rng, torch.get_rng_state())
    finally:
        method.close()
        method.close()
    assert "forward" not in model.__dict__
    assert all("forward" not in b.attn.__dict__ for b in model.blocks)
    assert not hasattr(model, "_evb_feature_cache")
    with torch.inference_mode():
        torch.testing.assert_close(
            model.forward_with_cfg(*inputs[0], 1.5),
            reference.forward_with_cfg(*inputs[0], 1.5),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "config", [DiffusionCacheConfig(enabled=False), DiffusionCacheConfig(cache_interval=1)]
)
def test_disabled_and_interval_one_leave_native_forward_untouched(config):
    model = native_dit()
    original = model.forward
    method = EfficientVLADiffusionCache(host_for(model), config)
    try:
        assert model.forward == original and not method.originals
    finally:
        method.close()


def test_geometry_error_invalidates_cache_and_training_is_rejected():
    model = native_dit()
    method = EfficientVLADiffusionCache(host_for(model), DiffusionCacheConfig())
    try:
        with torch.inference_mode():
            method.before_query({})
            model(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
            with pytest.raises(RuntimeError, match="geometry"):
                model(torch.zeros(4, 4, 7), torch.ones(4), torch.zeros(4, 1, 4096))
            assert not method.cache and not method.query_open
            method.before_query({})
            model(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
        with pytest.raises(RuntimeError, match="inference-only"):
            model(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
        assert not method.cache
    finally:
        method.close()


def test_config_and_backend_scope_are_explicit():
    from efficient_vla_benchmark.core.backend import BackendSpec
    from efficient_vla_benchmark.core.config import ExperimentConfig
    from efficient_vla_benchmark.core.registry import select_method

    for values in ({"cache_interval": True}, {"cache_interval": 0}, {"tokens_per_view": 128}):
        with pytest.raises(ValueError):
            DiffusionCacheConfig.parse(values)
    spec = BackendSpec("openvla_oft", "x:y", "x:z", ("libero_joint",), frozenset())
    with pytest.raises(ValueError, match="Unsupported"):
        select_method(
            ExperimentConfig(
                "openvla_oft", method="efficient_vla_diffusion_cache", model_variant="libero_joint"
            ),
            spec,
        )


@pytest.mark.gpu
def test_real_cogact_cache_counts_native_parity_reset_and_detach(model_path, trace_path, request):
    pytest.importorskip("efficient_vla_backend_cogact")
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    for config in (DiffusionCacheConfig(enabled=False), DiffusionCacheConfig(cache_interval=1)):
        host.attach_method(EfficientVLADiffusionCache(host, config))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        host.detach_method()
    method = EfficientVLADiffusionCache(host, DiffusionCacheConfig())
    host.attach_method(method)
    predictions = []
    for _ in range(2):
        host.reset(context)
        actions = host.predict_chunk(frames[0])
        assert actions.shape == (16, 7) and np.isfinite(actions).all()
        assert host.query_diagnostics["denoiser_calls"] == 10
        assert host.query_diagnostics["feature_refresh_steps"] == 2
        assert host.query_diagnostics["feature_reuse_steps"] == 8
        assert host.query_diagnostics["attention_evaluations"] == 24
        assert host.query_diagnostics["mlp_reuses"] == 96
        assert not method.cache
        predictions.append(actions)
    np.testing.assert_array_equal(*predictions)
    assert not np.array_equal(predictions[0], native)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
