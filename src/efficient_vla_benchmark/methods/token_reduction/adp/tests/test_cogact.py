"""CogACT ADP: reference scoring, single-view quotas and observed motion."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ..config import ADPConfig
from ..implementations.cogact.api import preflight
from ..scoring import embedding_scores
from .test_adp import reference_node

# Backend-specific tests must not make the optional CogACT plugin mandatory in OFT.
decoder = pytest.importorskip("efficient_vla_backend_cogact.decoder")
CausalTokenAdapter, TokenLayout = decoder.CausalTokenAdapter, decoder.TokenLayout
ADP = import_module(
    "efficient_vla_benchmark.methods.token_reduction.adp.implementations.cogact.method"
).ADP


def backend():
    torch.manual_seed(27)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
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


def test_camera_preflight_does_not_silently_drop_oft_weights():
    with pytest.raises(ValueError, match="one camera"):
        preflight(ADPConfig(), None)
    preflight(ADPConfig(view_weights=(1.0,)), None)


def test_all_nonvisual_inputs_and_selection_match_released_qk_rule():
    host, model = backend()
    embeddings = torch.randn(1, 12, 32)
    namespace = {"torch": torch, "PreTrainedModel": object}
    exec(
        reference_node(
            "prismatic/extern/hf/modeling_prismatic.py", "_compute_qk_importance_generic"
        ),
        namespace,
    )
    visual = embeddings[:, 1:5]
    text = torch.cat((embeddings[:, :1], embeddings[:, 5:]), dim=1)
    with torch.inference_mode():
        expected = namespace["_compute_qk_importance_generic"](model, text, visual, 0)
        torch.testing.assert_close(
            embedding_scores(model.model.layers[0], text, visual), expected, rtol=0, atol=0
        )
    method = ADP(host, ADPConfig(view_weights=(1.0,), dynamic=False))
    seen = []
    handle = model.model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["position_ids"].clone()),
        with_kwargs=True,
    )
    try:
        method.before_query({})
        with torch.inference_mode():
            output = model(inputs_embeds=embeddings, output_hidden_states=True)
        selected = sorted((expected.topk(3).indices + 1).tolist())
        assert seen[0][0].tolist() == [0, *selected, *range(5, 12)]
        assert output.hidden_states[-1].shape == (1, 11, 32)
        assert host.query_diagnostics["visual_tokens_final"] == 3
        assert host.query_diagnostics["layer_sequence_lengths"] == [11] * 3
    finally:
        handle.remove()
        method.close()


def test_dynamic_gate_uses_observations_and_resets_between_episodes():
    host, model = backend()
    method = ADP(host, ADPConfig(view_weights=(1.0,)))
    with pytest.raises(RuntimeError, match="EEF"):
        method.before_query({})
    results = []
    position = np.zeros(3)
    for window in range(8):
        method.observe(SimpleNamespace(proprio=position.copy()))
        method.before_query({})
        results.append(method.should_prune)
        for _ in range(8):
            position += [0.01 * (window + 1), 0, 0]
            method.observe(SimpleNamespace(proprio=position.copy()))
    assert results[:3] == [False] * 3
    assert any(results[3:])
    assert method.metadata()["extra_observation_inputs"] == ["observed_eef_xyz"]
    method.reset(None)
    assert method.gate.last is None and method.queries == 0
    method.close()


@pytest.mark.parametrize("parameters", [{"enabled": False}, {"keep_ratio": 1.0}])
def test_disabled_config_has_no_hooks_or_extra_state_requirement(parameters):
    host, model = backend()
    embeddings = torch.randn(1, 12, 32)
    with torch.inference_mode():
        expected = model(inputs_embeds=embeddings).logits
    method = ADP(host, ADPConfig.parse({"view_weights": [1.0], **parameters}))
    method.before_query({})
    assert method.adapter is None and not method.requires_control_history
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs_embeds=embeddings).logits, expected, rtol=0, atol=0)
    method.close()


@pytest.mark.gpu
def test_real_cogact_native_sparse_motion_reset_and_rng(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, observations, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(observations[0])
    for options in ({"enabled": False}, {"keep_ratio": 1.0}):
        host.attach_method(ADP(host, ADPConfig.parse({"view_weights": [1.0], **options})))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
        host.detach_method()

    host.attach_method(ADP(host, ADPConfig(view_weights=(1.0,), dynamic=False)))
    host.reset(context)
    torch_state = torch.cuda.get_rng_state().clone()
    sparse = host.predict_chunk(observations[0])
    assert np.isfinite(sparse).all() and sparse.shape == (16, 7)
    assert host.query_diagnostics["visual_tokens_final"] == 192
    sparse_rng = torch.cuda.get_rng_state().clone()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), sparse)
    host.detach_method()
    host.reset(context)
    assert torch.equal(torch.cuda.get_rng_state(), torch_state)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
    assert torch.equal(torch.cuda.get_rng_state(), sparse_rng)

    host.attach_method(ADP(host, ADPConfig(view_weights=(1.0,))))
    controller = ChunkPolicy(host, execution_horizon=8)
    controller.reset(context)
    actions, tokens, gates = [], [], []
    for observation in observations[:64]:
        decision = controller.step(observation)
        actions.append(decision.action.copy())
        if decision.queried:
            tokens.append(host.query_diagnostics["visual_tokens_final"])
            gates.append(host.query_diagnostics["pruned"])
    assert tokens[:3] == [256] * 3
    assert tokens == [192 if gate else 256 for gate in gates]
    controller.reset(context)
    replay = [controller.step(observation).action.copy() for observation in observations[:64]]
    np.testing.assert_array_equal(np.stack(replay), np.stack(actions))
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
