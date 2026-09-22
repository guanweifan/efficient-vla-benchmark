"""CogACT ICS checks against the actual attention output and an independent SVD."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ..config import FlashVLAConfig

# Backend-specific tests must not make the optional CogACT plugin mandatory in OFT.
decoder = pytest.importorskip("efficient_vla_backend_cogact.decoder")
CausalTokenAdapter, TokenLayout = decoder.CausalTokenAdapter, decoder.TokenLayout
FlashVLA = import_module(
    "efficient_vla_benchmark.methods.token_reduction.flash_vla.implementations.cogact.method"
).FlashVLA


def backend():
    torch.manual_seed(93)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    access = SimpleNamespace(
        decoder=model.model,
        patches_per_view=4,
        attach_token_strategy=lambda strategy: CausalTokenAdapter(
            model.model, strategy, lambda n: TokenLayout(n, patches_per_view=4)
        ),
    )
    return SimpleNamespace(model_access=access, query_diagnostics={}), model


def test_ics_selects_from_actual_attention_output_not_residual_or_input():
    host, model = backend()
    embeddings = torch.randn(1, 12, 32)
    method = FlashVLA(host, FlashVLAConfig(tokens_per_view=2))
    captured, positions = [], []
    handles = [
        model.model.layers[1].self_attn.o_proj.register_forward_hook(
            lambda module, args, output: captured.append(output[0, 1:5].clone())
        ),
        model.model.layers[2].register_forward_pre_hook(
            lambda module, args, kwargs: positions.append(kwargs["position_ids"].clone()),
            with_kwargs=True,
        ),
    ]
    try:
        method.before_query({})
        with torch.inference_mode():
            output = model(inputs_embeds=embeddings, output_hidden_states=True)
        left, values, _ = np.linalg.svd(captured[0].numpy(), full_matrices=False)
        score = np.abs(left * values).sum(-1)
        selected = sorted((np.argsort(-score, kind="stable")[:2] + 1).tolist())
        assert positions[0][0].tolist() == [0, *selected, *range(5, 12)]
        assert host.query_diagnostics["layer_sequence_lengths"] == [12, 12, 10, 10]
        assert host.query_diagnostics["visual_tokens_final"] == 2
        assert output.hidden_states[-1].shape == (1, 10, 32)
        assert method.attention_features is None
        assert not torch.equal(captured[0], embeddings[0, 1:5])
    finally:
        for handle in handles:
            handle.remove()
        method.close()


def test_disabled_and_full_retention_preserve_native_without_hooks():
    host, model = backend()
    embeddings = torch.randn(1, 12, 32)
    with torch.inference_mode():
        expected = model(inputs_embeds=embeddings).logits
    for config in (FlashVLAConfig(enabled=False), FlashVLAConfig(tokens_per_view=4)):
        method = FlashVLA(host, config)
        assert method.adapter is None and method.handle is None
        method.before_query({})
        with torch.inference_mode():
            torch.testing.assert_close(
                model(inputs_embeds=embeddings).logits, expected, rtol=0, atol=0
            )
        method.close()


def test_causal_visual_ics_does_not_gain_future_language_context():
    host, model = backend()
    embeddings = torch.randn(1, 12, 32)
    changed = embeddings.clone()
    changed[:, 5:] *= -7
    method = FlashVLA(host, FlashVLAConfig(tokens_per_view=2))
    selected = []
    handle = model.model.layers[2].register_forward_pre_hook(
        lambda module, args, kwargs: selected.append(kwargs["position_ids"].clone()),
        with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            method.before_query({})
            first = model(inputs_embeds=embeddings).logits
            method.reset(None)
            method.before_query({})
            second = model(inputs_embeds=changed).logits
        torch.testing.assert_close(selected[0], selected[1], rtol=0, atol=0)
        assert not torch.equal(first[:, -1], second[:, -1])
    finally:
        handle.remove()
        method.close()


@pytest.mark.gpu
def test_real_cogact_ics_parity_selection_reset_and_rng(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, observations, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(observations[0])
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (FlashVLAConfig(enabled=False), FlashVLAConfig(tokens_per_view=256)):
        host.attach_method(FlashVLA(host, config))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
        host.detach_method()
    host.attach_method(FlashVLA(host, FlashVLAConfig()))
    host.reset(context)
    sparse = host.predict_chunk(observations[0])
    assert sparse.shape == (16, 7) and np.isfinite(sparse).all()
    assert host.query_diagnostics["visual_tokens_final"] == 160
    lengths = host.query_diagnostics["layer_sequence_lengths"]
    assert lengths[:2] == [lengths[0]] * 2
    assert lengths[2:] == [lengths[0] - 96] * 30
    assert host.method.attention_features is None
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), sparse)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
