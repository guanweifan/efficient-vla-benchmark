"""Full shared policy with native causal cognition and immutable skipped KV."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from ..config import VLACacheConfig


@torch.inference_mode()
def test_complete_causal_cache_and_instruction_invalidation():
    pytest.importorskip("efficient_vla_backend_cogact")
    from efficient_vla_backend_cogact.reuse import CausalReuseAdapter

    from ..implementations.cogact.method import VLACache

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(9)
    decoder = LlamaModel(config).eval()
    data = torch.randn(1, 300, 16)
    native = decoder(inputs_embeds=data).last_hidden_state
    access = SimpleNamespace(
        decoder=decoder,
        policy_image=lambda x: x,
        attach_reuse_strategy=lambda s: CausalReuseAdapter(decoder, s),
    )
    backend = SimpleNamespace(model_access=access, device=torch.device("cpu"), query_diagnostics={})
    method = VLACache(backend, VLACacheConfig())
    prepared = {
        "image": Image.fromarray(np.full((224, 224, 3), 100, np.uint8)),
        "instruction": "pick cup",
    }
    try:
        method.before_query(prepared)
        cold = decoder(inputs_embeds=data).last_hidden_state
        method.after_query(None)
        torch.testing.assert_close(cold, native, atol=1e-6, rtol=0)
        assert len(method.previous_schedule) == 32 and len(method.previous_protected[0]) == 100
        method.before_query(prepared)
        warm = decoder(inputs_embeds=data).last_hidden_state
        method.after_query(None)
        assert backend.query_diagnostics["reused_tokens_last_layer"] > 0
        torch.testing.assert_close(warm[:, -1], native[:, -1], atol=1e-6, rtol=0)
        assert all(k.shape[2] == 300 for k in method.store.keys.values())
        assert method.last_positions[-1] == 299
        method.before_query({**prepared, "instruction": "place cup"})
        assert not method.store.keys
        decoder(inputs_embeds=data)
        assert backend.query_diagnostics["cache_cold"]
        method.reset(None)
        assert not method.store.keys and method.previous_images is None
    finally:
        method.close()
    torch.testing.assert_close(
        decoder(inputs_embeds=data).last_hidden_state, native, atol=0, rtol=0
    )


@pytest.mark.gpu
@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_real_causal_cache_reference_reuse_rng_and_reset(
    model_path, trace_path, attention_backend, request
):
    from dataclasses import replace
    from types import MethodType

    from efficient_vla_backend_cogact.backend import CogACTBackend
    from transformers.models.llama.modeling_llama import LlamaAttention

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import VLACache

    context, samples, _ = load_trace(trace_path, require_control_steps=True)
    backend = CogACTBackend(model_path)
    request.addfinalizer(backend.close)
    decoder = backend.model_access.decoder
    backend.reset(context)
    native = backend.predict_chunk(samples[0])
    native_rng = torch.cuda.get_rng_state().clone()
    reference = native
    if attention_backend == "eager":
        originals = [(layer.self_attn, layer.self_attn.forward) for layer in decoder.layers]
        try:
            decoder.config._attn_implementation = "eager"
            for attention, _ in originals:
                attention.forward = MethodType(LlamaAttention.forward, attention)
            backend.reset(context)
            reference = backend.predict_chunk(samples[0])
        finally:
            decoder.config._attn_implementation = "sdpa"
            for attention, forward in originals:
                attention.forward = forward
    method = VLACache(backend, VLACacheConfig(attention_backend=attention_backend))
    backend.attach_method(method)
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), reference, atol=1e-6, rtol=0)
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    assert backend.metadata()["attention_implementation"] == attention_backend
    assert backend.query_diagnostics["cache_cold"]
    reused = 0
    for sample in samples[8:40:8]:
        saved = method.store.keys[31].clone()
        actions = backend.predict_chunk(sample)
        assert actions.shape == (16, 7) and np.isfinite(actions).all()
        dropped = method.full_positions[~torch.isin(method.full_positions, method.last_positions)]
        if dropped.numel():
            torch.testing.assert_close(
                method.store.keys[31][:, :, dropped], saved[:, :, dropped], atol=0, rtol=0
            )
            reused += dropped.numel()
        assert len(method.previous_schedule) == 32
        assert len(method.previous_protected) == 1 and len(method.previous_protected[0]) == 100
        assert method.last_positions[-1] == method.last_layout.cognition_position
        assert all(k.shape[2] == method.last_layout.length for k in method.store.keys.values())
    assert reused > 0
    backend.predict_chunk(replace(samples[0], instruction="pick up the object carefully"))
    assert backend.query_diagnostics["cache_cold"]
    backend.reset(context)
    assert not method.store.keys and method.previous_images is None
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), reference, atol=1e-6, rtol=0)
    backend.detach_method()
    assert backend.metadata()["attention_implementation"] == "sdpa"
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), native)
