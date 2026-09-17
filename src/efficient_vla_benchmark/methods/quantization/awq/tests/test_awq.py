import json
from dataclasses import replace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from efficient_vla_benchmark.methods.quantization.awq.config import AWQConfig

from ..implementations.openvla_oft.artifact import (
    read_manifest,
)
from ..implementations.openvla_oft.search import (
    apply_layer_scales,
    batch_inputs,
    layer_arguments,
    search_scale,
)


@pytest.mark.parametrize(
    "parameters",
    [
        {"bits": 8},
        {"group_size": 16},
        {"calibration_samples": 5},
        {"clip_grid": 0},
        {"clip_max_shrink": float("nan")},
        {"calibration_seed": -1},
        {"unknown": 1},
        {"kernel_precision": "int8"},
    ],
)
def test_awq_configuration_validation(parameters):
    with pytest.raises(ValueError):
        AWQConfig.parse(parameters)


def tiny_layer():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
    )
    config._attn_implementation = "sdpa"
    return LlamaDecoderLayer(config, 0).eval()


@torch.inference_mode()
def test_calibration_padding_preserves_independent_observations():
    torch.manual_seed(11)
    layer = tiny_layer()
    samples = [torch.randn(1, length, 64) for length in (12, 9)]
    inputs, valid = batch_inputs(samples, "cpu", torch.float32)
    actual = layer(inputs, **layer_arguments(valid, inputs.dtype))[0]
    for index, sample in enumerate(samples):
        positions = torch.arange(sample.shape[1]).unsqueeze(0)
        expected = layer(sample, position_ids=positions, use_cache=False)[0]
        torch.testing.assert_close(actual[index, : sample.shape[1]], expected[0], atol=1e-6, rtol=0)


@torch.inference_mode()
def test_awq_scale_transfer_preserves_unquantized_layer():
    torch.manual_seed(12)
    layer = tiny_layer()
    inputs, valid = batch_inputs(
        [torch.randn(1, 12, 64), torch.randn(1, 9, 64)], "cpu", torch.float32
    )
    kwargs = layer_arguments(valid, inputs.dtype)
    expected = layer(inputs, **kwargs)[0]
    scales = [
        (
            "input_layernorm",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            torch.rand(64) + 0.5,
        ),
        ("self_attn.v_proj", ("self_attn.o_proj",), torch.rand(64) + 0.5),
        ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), torch.rand(64) + 0.5),
        ("mlp.up_proj", ("mlp.down_proj",), torch.rand(128) + 0.5),
    ]
    apply_layer_scales(layer, scales)
    actual = layer(inputs, **kwargs)[0]
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-6, rtol=0)


@torch.inference_mode()
def test_scale_search_restores_weights_and_ignores_padding():
    torch.manual_seed(13)
    linear = torch.nn.Linear(64, 64, bias=False)
    original = linear.weight.clone()
    inputs, valid = batch_inputs(
        [torch.randn(1, 8, 64), torch.randn(1, 5, 64)], "cpu", torch.float32
    )
    config = AWQConfig.parse({"group_size": 32, "scale_grid": 4})
    scale_a, stats_a = search_scale(linear, [linear], inputs, valid, {}, config)
    inputs[~valid] = 10000
    scale_b, stats_b = search_scale(linear, [linear], inputs, valid, {}, config)
    torch.testing.assert_close(scale_a, scale_b, atol=0, rtol=0)
    assert stats_a == stats_b
    assert stats_a["mse"] <= stats_a["candidate_mse"][0]
    torch.testing.assert_close(linear.weight, original, atol=0, rtol=0)


def test_artifact_rejects_incomplete_and_path_traversal(tmp_path):
    config = AWQConfig.parse({})
    with pytest.raises(ValueError, match="completed"):
        read_manifest(tmp_path, config)
    manifest = {
        "schema_version": 1,
        "algorithm": "awq",
        "complete": True,
        "config": config.calibration_parameters(),
        "scope": "language_decoder_block_linears",
        "components": [{"file": f"layer_{index:02d}.safetensors"} for index in range(32)],
    }
    manifest["components"][0]["file"] = "../unexpected.safetensors"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        read_manifest(tmp_path, config)


def test_runtime_precision_does_not_change_calibration_identity():
    config = AWQConfig.parse({})
    runtime = replace(config, kernel_precision="float16")
    assert config.calibration_parameters() == runtime.calibration_parameters()
    assert replace(config, scale_grid=3).calibration_parameters() != config.calibration_parameters()


@pytest.mark.parametrize("group_size", [32, 64])
def test_runtime_rejects_groups_not_supported_by_pinned_gemm(group_size):
    from ..implementations.openvla_oft.api import preflight

    with pytest.raises(ValueError, match="group_size=128"):
        preflight(AWQConfig.parse({"group_size": group_size}), None)
