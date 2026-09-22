"""Full-context causal KV reuse with the paper's image/text/time selector."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ..config import InfoEntropyConfig

access_module = pytest.importorskip("efficient_vla_backend_cogact.reuse")
VLAInfoEntropy = import_module(
    "efficient_vla_benchmark.methods.caching.vla_infoentropy.implementations.cogact.method"
).VLAInfoEntropy


def test_causal_cache_keeps_context_protects_text_and_resets():
    torch.manual_seed(24)
    cfg = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=12,
        num_attention_heads=2,
    )
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).eval()
    access = SimpleNamespace(
        decoder=model.model,
        policy_image=np.asarray,
        attach_reuse_strategy=lambda strategy: access_module.CausalReuseAdapter(
            model.model, strategy
        ),
    )
    host = SimpleNamespace(model_access=access, device=torch.device("cpu"), query_diagnostics={})
    image = np.random.default_rng(11).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    prepared = {"image": image, "instruction": "pick"}
    hidden = torch.randn(1, 264, 16)
    with torch.inference_mode():
        native = model(inputs_embeds=hidden).logits
    method = VLAInfoEntropy(host, InfoEntropyConfig(attention_backend="sdpa"))
    try:
        with torch.inference_mode():
            method.before_query(prepared)
            torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
            method.after_query(None)
            assert method.previous_information.shape == (1, 256)
            assert len(method.previous_schedule) == 12
            method.before_plan(None, 50)
            method.before_query(prepared)
            assert host.query_diagnostics["visual_quota"] == 50
            assert method.reusable.numel() > 0
            model(inputs_embeds=hidden)
            method.after_query(None)
            assert host.query_diagnostics["visual_kv_tokens"] == 256
            assert host.query_diagnostics["visual_query_tokens_final"] < 256
            assert all(k.shape[2] == 264 for k in method.store.keys.values())
            assert not host.query_diagnostics["cache_cold"]
            method.before_query(dict(prepared, instruction="place"))
            assert not method.store.keys and method.step == 0
            torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
            method.abort()
            assert not method.store.keys and method.previous_information is None
            method.reset(None)
            assert method.step == 0 and method.last_instruction is None
    finally:
        method.close()
        method.close()
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
    assert not hasattr(model.model, "_evb_token_adapter")


@pytest.mark.gpu
def test_real_cogact_native_parity_warm_kv_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    for cfg in (InfoEntropyConfig(enabled=False), InfoEntropyConfig(static_budget_per_view=0)):
        host.attach_method(VLAInfoEntropy(host, cfg))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        host.detach_method()
    method = VLAInfoEntropy(host, InfoEntropyConfig(attention_backend="sdpa"))
    host.attach_method(method)
    host.reset(context)
    np.testing.assert_allclose(host.predict_chunk(frames[0]), native, rtol=0, atol=1e-6)
    assert host.query_diagnostics["cache_cold"]
    method.before_plan(frames[0], 50)
    warm = host.predict_chunk(frames[0])
    assert warm.shape == (16, 7) and np.isfinite(warm).all()
    assert host.query_diagnostics["visual_query_tokens_final"] < 256
    assert host.query_diagnostics["visual_kv_tokens"] == 256
    assert host.query_diagnostics["visual_quota"] == 50
    assert len(method.store.keys) == 32
    host.reset(context)
    np.testing.assert_allclose(host.predict_chunk(frames[0]), native, rtol=0, atol=1e-6)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
