"""Single-view native causal pruning with complete temporal/controller machinery."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import LlamaConfig, LlamaForCausalLM

from ..config import SpecPruneConfig

decoder_module = pytest.importorskip("efficient_vla_backend_cogact.decoder")
SpecPrune = import_module(
    "efficient_vla_benchmark.methods.token_reduction.specprune_vla.implementations.cogact.method"
).SpecPrune


def test_native_sparse_layers_global_history_controller_and_noop():
    torch.manual_seed(7)
    cfg = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
    )
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).eval()

    def native_policy_image(image):
        assert isinstance(image, Image.Image)
        return np.asarray(image)

    access = SimpleNamespace(
        decoder=model.model,
        policy_image=native_policy_image,
        attach_token_strategy=lambda s: decoder_module.CausalTokenAdapter(model.model, s),
    )
    host = SimpleNamespace(model_access=access, query_diagnostics={})
    hidden = torch.randn(1, 282, 16)
    image = np.random.default_rng(2).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    prepared = {"image": image, "instruction": "pick up"}
    with torch.inference_mode():
        native = model(inputs_embeds=hidden).logits
    disabled = SpecPrune(
        host, SpecPruneConfig(static_pruning=False, dynamic_pruning=False, action_controller=False)
    )
    try:
        with torch.inference_mode():
            disabled.before_query(prepared)
            torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)
    finally:
        disabled.close()
    # Moderate static budget exercises every dynamic stage independently of
    # whether the aggressive benchmark preset reaches its minimum early.
    method = SpecPrune(host, SpecPruneConfig(primary_static_budget=128, min_visual_tokens=30))
    selected = []
    handle = model.model.layers[-1].register_forward_pre_hook(
        lambda m, a, kw: selected.append(kw["position_ids"].clone()), with_kwargs=True
    )
    try:
        for _ in range(2):
            method.observe(SimpleNamespace(image=image))
            method.before_query(prepared)
            with torch.inference_mode():
                model(inputs_embeds=hidden)
            assert 0 < host.query_diagnostics["final_visual_tokens"] < 256
            assert host.query_diagnostics["importance_updates"] == [14, 19, 24]
            assert host.query_diagnostics["dynamic_pruning_layers"] == [10, 15, 20, 25]
            assert method.previous_global.numel() > 0
        assert selected[-1][0, -1] == 281
        assert selected[-1][0, -25:].tolist() == list(range(257, 282))
        method.control_steps = 11
        fast = np.array([0.7, 0, 0, 0, 0, 0, -1.0])
        slow = np.array([0.01, 0, 0, 0, 0, 0, -1.0])
        assert not method.after_action(fast, 5)
        assert method.after_action(slow, 4)
        assert method.precise and method.replans == 1
        method.reset(None)
        assert method.confidence == {} and method.previous_global is None and not method.frames
    finally:
        handle.remove()
        method.close()
        method.close()
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs_embeds=hidden).logits, native, rtol=0, atol=0)


@pytest.mark.gpu
def test_real_cogact_noop_all_mechanisms_and_replay_reset(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    host.attach_method(
        SpecPrune(
            host,
            SpecPruneConfig(static_pruning=False, dynamic_pruning=False, action_controller=False),
        )
    )
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
    host.detach_method()
    method = SpecPrune(host, SpecPruneConfig(min_visual_tokens=30))
    host.attach_method(method)
    policy = ChunkPolicy(host, 8)
    commands = []
    for _ in range(2):
        policy.reset(context)
        current, diagnostics = [], []
        for frame in frames[:32]:
            decision = policy.step(frame)
            current.append(decision.action)
            assert np.isfinite(decision.action).all()
            if decision.queried:
                diagnostics.append(dict(host.query_diagnostics))
        assert all(0 < d["final_visual_tokens"] < 256 for d in diagnostics)
        assert any(d["global_reuse_tokens"] for d in diagnostics)
        assert any(d["dynamic_pruning_layers"] for d in diagnostics)
        assert any(d["importance_updates"] for d in diagnostics)
        commands.append(np.stack(current))
    np.testing.assert_array_equal(commands[0], commands[1])
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
