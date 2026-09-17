import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderState, TokenLayout
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from efficient_vla_benchmark.core.resources import project_source
from efficient_vla_benchmark.core.types import EpisodeContext
from efficient_vla_benchmark.methods.token_reduction.specprune_vla.config import SpecPruneConfig

from ..implementations.openvla_oft.method import (
    SpecPrune,
    stable_patch_ids,
)


def method(parameters=None):
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    cfg._attn_implementation = "sdpa"
    decoder = LlamaModel(cfg).eval()
    backend = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(model=decoder)),
        query_diagnostics={},
    )
    return SpecPrune(backend, SpecPruneConfig.parse(parameters or {}))


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 1},
        {"static_ratio": float("nan")},
        {"action_controller": "yes"},
        {"pruning_layers": [10, 10]},
        {"importance_layers": [-1]},
        {"primary_static_budget": -1},
        {"min_visual_tokens": 0},
    ],
)
def test_invalid_specprune_settings_fail_before_loading(parameters):
    with pytest.raises(ValueError):
        SpecPruneConfig.parse(parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"importance_layers": [32]},
        {"primary_static_budget": 257},
        {"min_visual_tokens": 513},
    ],
)
def test_model_bounds_are_validated_by_oft_implementation(parameters):
    from ..implementations.openvla_oft.api import preflight

    config = SpecPruneConfig.parse(parameters)
    with pytest.raises(ValueError, match="OFT"):
        preflight(config, None)


def test_patch_selection_matches_pinned_reference():
    source = project_source("third_party/methods/specprune_vla") / (
        "openvla-oft/experiments/robot/spec_prune_vla.py"
    )
    spec = importlib.util.spec_from_file_location("reference_patch_similarity", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(3)
    image = rng.integers(20, 240, size=(224, 224, 3), dtype=np.uint8)
    previous = image.copy()
    previous[14:28, 28:42] = 0
    for view, offset in (("primary", 1), ("wrist", 257)):
        expected = reference.get_similarity_indices(
            image, previous, 230, sim_threshold=0.98, view=view
        )
        actual = stable_patch_ids(image, previous, 230, 0.98, offset)
        np.testing.assert_array_equal(np.sort(actual), np.sort(expected))
    assert stable_patch_ids(image, previous, 0, 0.98, 1).size == 0


def test_static_selection_protects_prior_global_and_nonvisual_tokens():
    strategy = method({"dynamic_pruning": False, "action_controller": False})
    state = DecoderState(TokenLayout(600), torch.arange(600))
    strategy.last_layout = state.layout
    strategy.previous_global = torch.tensor([10, 270])
    strategy.stable_ids = np.arange(1, 513)
    strategy.start(state)
    scores = torch.arange(600).float().expand(4, -1)
    keep = strategy.after_layer(0, state, {"local_text": scores})
    assert keep[10] and keep[270] and keep[0] and keep[513:].all()
    assert keep.sum() < 600
    state.positions = state.positions[keep]
    keep = strategy.after_layer(1, state, {"local_text": scores[:, state.positions]})
    kept = set(state.positions[keep].tolist())
    assert {10, 270}.issubset(kept)
    assert set(range(513, 600)).issubset(kept)
    strategy.close()


def test_dynamic_pruning_keeps_budget_protected_tokens_and_episode_confidence():
    strategy = method({"static_pruning": False, "action_controller": False})
    state = DecoderState(TokenLayout(600), torch.arange(600))
    strategy.start(state)
    attention = torch.rand(56, 600)
    strategy.after_layer(14, state, {"actions": attention})
    old_confidence = strategy.confidence[14]
    strategy.after_layer(14, state, {"actions": attention.flip(1)})
    assert strategy.confidence[14] is old_confidence
    keep = strategy.after_layer(15, state, {})
    assert keep.sum() == 540
    assert keep[0] and keep[513:].all()
    assert torch.count_nonzero(strategy.scores[1:513]) > 0
    strategy.reset(EpisodeContext("libero_goal", 0, 1))
    assert strategy.confidence == {} and strategy.previous_global is None
    strategy.close()


def test_controller_requests_replan_on_fine_motion_transition_and_resets():
    strategy = method()
    strategy.control_steps = 11
    fast = np.array([0.7, 0, 0, 0, 0, 0, -1.0])
    assert not strategy.after_action(fast, remaining=5)
    assert not strategy.precise
    slow = np.array([0.01, 0, 0, 0, 0, 0, -1.0])
    assert strategy.after_action(slow, remaining=4)
    assert strategy.precise and strategy.replans == 1
    vertical = np.array([0, 0, 0.4, 0, 0, 0, -1.0])
    assert not strategy.after_action(vertical, remaining=2)
    assert strategy.precise
    assert not strategy.after_action(vertical, remaining=0)
    assert not strategy.precise
    strategy.reset(EpisodeContext("libero_goal", 0, 1))
    assert strategy.control_steps == strategy.replans == 0
    assert not strategy.precise and len(strategy.frames) == 0
    strategy.close()


@torch.inference_mode()
def test_disabled_internal_mechanisms_are_dense_equivalent():
    strategy = method(
        {"static_pruning": False, "dynamic_pruning": False, "action_controller": False}
    )
    decoder = strategy.backend.model.language_model.model
    data = torch.randn(1, 600, 16)
    actual = decoder(inputs_embeds=data).last_hidden_state
    strategy.close()
    reference = decoder(inputs_embeds=data).last_hidden_state
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=0)
