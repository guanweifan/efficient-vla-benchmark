import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from efficient_vla_benchmark.core.resources import project_source
from efficient_vla_benchmark.methods.caching.vla_cache.config import VLACacheConfig

from ..selection import reuse_schedule, static_patch_ids


def reference_utils():
    pytest.importorskip("skimage.util", reason="Pinned author reference needs scikit-image")
    source = project_source("third_party/methods/vla_cache") / (
        "src/openvla-oft/experiments/robot/vla_cache_utils.py"
    )
    spec = importlib.util.spec_from_file_location("reference_vla_cache", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 1},
        {"similarity_threshold": 2},
        {"growth_factor": float("nan")},
        {"reuse_layers": [6, 2]},
        {"task_attention_layer": -1},
        {"weighted_growth": 1},
        {"protected_tokens_per_view": -1},
        {"attention_backend": "unknown"},
    ],
)
def test_cache_config_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        VLACacheConfig.parse(parameters)


def test_static_patch_order_matches_reference():
    rng = np.random.default_rng(4)
    current = rng.integers(20, 240, size=(224, 224, 3), dtype=np.uint8)
    previous = current.copy()
    previous[:70, :70] = 0
    expected = reference_utils().find_static_patches(current, previous)
    actual = static_patch_ids(current, previous, 150, 0.996)
    assert actual.tolist() == expected
    assert static_patch_ids(current, previous, 0, 0.996).size == 0


def test_entropy_schedule_matches_reference():
    torch.manual_seed(4)
    maps = [
        torch.softmax(torch.randn(1, 4, 12, 12) * (index + 1) / 8, dim=-1) for index in range(32)
    ]
    expected = reference_utils().get_layer_mask_schedule((*maps, torch.arange(12)))
    entropies = []
    for attention in maps:
        mean = attention.mean(dim=1)[0]
        mean = mean / (mean.sum(dim=-1, keepdim=True) + 1e-10)
        entropies.append(-(mean * (mean + 1e-10).log()).sum(dim=-1).mean())
    actual = torch.tensor(reuse_schedule(entropies))
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
    assert reuse_schedule([torch.tensor(1.0)] * 32) == [1.0] * 32


@torch.inference_mode()
def test_complete_cache_cold_parity_warm_reuse_and_language_invalidation():
    pytest.importorskip("efficient_vla_backend_openvla_oft")
    from ..implementations.openvla_oft.method import VLACache

    torch.manual_seed(9)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "sdpa"
    decoder = LlamaModel(config).eval()
    data = torch.randn(1, 600, 16)
    reference = decoder(inputs_embeds=data).last_hidden_state
    backend = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(model=decoder)),
        query_diagnostics={},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    method = VLACache(backend, VLACacheConfig.parse({}))
    image = np.full((224, 224, 3), 100, dtype=np.uint8)
    prepared = {
        "observation": SimpleNamespace(instruction="pick up object"),
        "images": [image, image],
    }
    method.before_query(prepared)
    cold = decoder(inputs_embeds=data).last_hidden_state
    method.after_query(None)
    torch.testing.assert_close(cold, reference, atol=1e-6, rtol=0)
    assert backend.query_diagnostics["cache_cold"]
    method.before_query(prepared)
    warm = decoder(inputs_embeds=data).last_hidden_state
    method.after_query(None)
    assert not backend.query_diagnostics["cache_cold"]
    assert backend.query_diagnostics["reused_tokens_last_layer"] > 0
    torch.testing.assert_close(warm[:, 513:], reference[:, 513:], atol=1e-6, rtol=0)
    assert all(value.shape[2] == 600 for value in method.store.keys.values())
    prepared["observation"].instruction = "different instruction"
    method.before_query(prepared)
    assert method.store.keys == {} and method.previous_images is None
    decoder(inputs_embeds=data)
    assert backend.query_diagnostics["cache_cold"]
    method.reset(None)
    assert method.store.keys == {} and method.previous_schedule is None
    method.close()
