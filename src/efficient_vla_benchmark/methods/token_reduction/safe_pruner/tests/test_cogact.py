"""Native-cognition forecast, refresh, protected readout and lifecycle checks."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ..config import SAFEPrunerConfig

decoder = pytest.importorskip("efficient_vla_backend_cogact.decoder")
SAFEPruner = import_module(
    "efficient_vla_benchmark.methods.token_reduction.safe_pruner.implementations.cogact.method"
).SAFEPruner


def test_dense_forecast_sparse_refresh_reset_and_detach():
    torch.manual_seed(21)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
    )
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).eval()
    access = SimpleNamespace(
        decoder=model.model,
        patches_per_view=8,
        attach_token_strategy=lambda strategy: decoder.CausalTokenAdapter(
            model.model, strategy, lambda n: decoder.TokenLayout(n, patches_per_view=8)
        ),
    )
    host = SimpleNamespace(model_access=access, query_diagnostics={})
    config = SAFEPrunerConfig(reference_layers=(0, 3, 4), keep_ratio=0.5)
    hidden = torch.randn(1, 16, 32)
    with torch.inference_mode():
        native = model(inputs_embeds=hidden).logits
    method = SAFEPruner(host, config)
    positions = []
    handle = model.model.layers[3].register_forward_pre_hook(
        lambda m, a, kw: positions.append(kw["position_ids"].clone()), with_kwargs=True
    )
    try:
        with torch.inference_mode():
            method.before_query({})
            torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
            assert host.query_diagnostics["keyframe"]
            assert method.reference_features.shape == (8, 32)
            assert method.reference_saliency.shape == (3, 8)
            reference = method.reference_saliency.clone()
            method.before_query({})
            sparse = model(inputs_embeds=hidden).logits
            assert sparse.shape == (1, 12, 32)
            assert host.query_diagnostics["visual_tokens_final"] == 4
            assert not host.query_diagnostics["keyframe"]
            torch.testing.assert_close(method.reference_saliency, reference)
            assert positions[-1][0, -1] == 15
            assert positions[-1][0, -7:].tolist() == list(range(9, 16))
            method.reference_saliency.zero_()
            method.before_query({})
            torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
            assert host.query_diagnostics["refreshes"] == 2
            method.reset(None)
            assert method.reference_features is None
            method.before_query({})
            torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
    finally:
        handle.remove()
        method.close()
        method.close()
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
    assert not hasattr(model.model, "_evb_token_adapter")


@pytest.mark.gpu
def test_real_cogact_forecast_native_parity_and_rng(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    for cfg in (SAFEPrunerConfig(enabled=False), SAFEPrunerConfig(keep_ratio=1)):
        host.attach_method(SAFEPruner(host, cfg))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        host.detach_method()
    method = SAFEPruner(host, SAFEPrunerConfig(reference_layers=(0, 30, 31)))
    host.attach_method(method)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    assert method.reference_saliency.shape == (3, 256)
    sparse = host.predict_chunk(frames[0])
    assert sparse.shape == (16, 7) and np.isfinite(sparse).all()
    assert not host.query_diagnostics["keyframe"]
    assert host.query_diagnostics["visual_tokens_final"] == 64
    assert method.current_features is None
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), sparse)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
