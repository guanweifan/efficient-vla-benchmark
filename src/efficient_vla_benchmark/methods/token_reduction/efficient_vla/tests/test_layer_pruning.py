import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from ..layer_config import LayerPruningConfig
from ..layer_pruning import LayerPruning, ranking, read_artifact


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError, match="finite"):
        ranking(scores)


def test_ranking_is_noncontiguous_and_stable():
    assert ranking([0.2, 0.1, 0.8, 0.1]) == [1, 3, 0, 2]


@pytest.mark.parametrize("config", [{"prune_layers": -1}, {"calibration_samples": 3}])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        LayerPruningConfig.parse(config)


@torch.inference_mode()
def test_physical_compaction_matches_explicit_layer_removal_and_restores(tmp_path):
    torch.manual_seed(9)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    config._attn_implementation = "sdpa"
    decoder = LlamaModel(config).eval()
    data = torch.randn(1, 12, 32)
    native = decoder(inputs_embeds=data, use_cache=False).last_hidden_state.clone()
    original = decoder.layers
    original_parameters = {name: p.clone() for name, p in decoder.named_parameters()}
    cfg = LayerPruningConfig(prune_layers=2, calibration_samples=4)
    manifest = {
        "schema_version": 1,
        "algorithm": "efficient_vla_layer_pruning",
        "complete": True,
        "backend": "test",
        "checkpoint_sha256": "test",
        "precision": "torch.float32",
        "num_layers": 4,
        "scores": [0.2, 0.1, 0.8, 0.1],
        "samples": [{"split": "train"}] * 4,
        "calibration_parameters": cfg.calibration_parameters(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    backend = SimpleNamespace(_model_manifest={"sha256": "test"})
    method = LayerPruning(backend, decoder, cfg, SimpleNamespace(artifact_path=tmp_path), "test")
    assert method.removed == [1, 3]
    assert len(decoder.layers) == decoder.config.num_hidden_layers == 2
    assert decoder.layers[0] is original[0] and decoder.layers[1] is original[2]
    assert [layer.self_attn.layer_idx for layer in decoder.layers] == [0, 1]
    pruned = decoder(inputs_embeds=data, use_cache=True).last_hidden_state
    # Independent direct recurrence over only the survivors.
    positions = torch.arange(data.shape[1])[None]
    hidden = data
    for layer in (original[0], original[2]):
        hidden = layer(hidden, position_ids=positions, use_cache=False)[0]
    torch.testing.assert_close(pruned, decoder.norm(hidden), rtol=0, atol=1e-6)
    assert not torch.equal(pruned, native)
    method.close()
    method.close()
    assert decoder.layers is original and config.num_hidden_layers == 4
    assert [layer.self_attn.layer_idx for layer in original] == [0, 1, 2, 3]
    for name, p in decoder.named_parameters():
        torch.testing.assert_close(p, original_parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(
        decoder(inputs_embeds=data, use_cache=False).last_hidden_state, native, rtol=0, atol=0
    )
    manifest["backend"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible"):
        read_artifact(tmp_path, cfg, "test")
