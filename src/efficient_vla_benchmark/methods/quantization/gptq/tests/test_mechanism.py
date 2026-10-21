import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.evaluation.libero import SUITES
from efficient_vla_benchmark.methods.quantization.gptq.artifact import read_manifest
from efficient_vla_benchmark.methods.quantization.gptq.calibrate import quantize_layer
from efficient_vla_benchmark.methods.quantization.gptq.config import GPTQConfig
from efficient_vla_benchmark.methods.quantization.gptq.mechanism import (
    CONTRACT,
    PROJECTIONS,
    capture_inputs,
    native_arguments,
    validate_layer,
)
from efficient_vla_benchmark.model_ops.quantization.packed import pack_signed, unpack_signed


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = nn.Module(), nn.Module()
        for name in PROJECTIONS:
            parent, attr = name.split(".")
            setattr(
                getattr(self, parent), attr, nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
            )

    def forward(self, hidden, **kwargs):
        attention = self.self_attn
        q, k, v = attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)
        hidden = hidden + attention.o_proj((q + k + v) / 3)
        hidden = hidden + self.mlp.down_proj(
            torch.nn.functional.silu(self.mlp.gate_proj(hidden)) * self.mlp.up_proj(hidden)
        )
        return (hidden,)


@pytest.mark.parametrize(
    "parameters",
    [
        dict(unknown=1),
        dict(calibration_samples=5),
        dict(calibration_samples=True),
        dict(calibration_seed=-1),
        dict(damp_percent=0),
        dict(damp_percent=float("nan")),
        dict(block_size=127),
        dict(block_size=True),
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        GPTQConfig.parse(parameters)


def test_native_masks_positions_cache_rejection():
    mask = torch.tensor([[[[0.0, -float("inf")], [0.0, 0.0]]]])
    positions = torch.arange(2)[None]
    values = native_arguments(
        dict(
            attention_mask=mask,
            position_ids=positions,
            past_key_value=SimpleNamespace(get_seq_length=lambda: 0),
            use_cache=True,
        )
    )
    assert torch.equal(values["attention_mask"], mask)
    assert torch.equal(values["position_ids"], positions)
    assert values["past_key_value"] is None and values["use_cache"] is False
    with pytest.raises(ValueError):
        native_arguments(dict(past_key_value=SimpleNamespace(get_seq_length=lambda: 1)))
    with pytest.raises(ValueError):
        native_arguments(dict(unverified_argument=1))


def test_capture_restores_hooks_on_success_and_failure():
    layer = Layer()

    def run(obs):
        return layer(obs, position_ids=torch.arange(obs.shape[1])[None])[0]

    backend = SimpleNamespace(reset=lambda context: None, predict_chunk=run)
    hidden = torch.randn(1, 3, 128).bfloat16()
    samples, identities = capture_inputs(backend, [(None, hidden, {"id": 1})], [layer])
    assert len(samples) == 1 and identities == [{"id": 1}]
    assert torch.equal(samples[0][0], hidden) and not layer._forward_pre_hooks
    backend.predict_chunk = lambda obs: None
    with pytest.raises(ValueError):
        capture_inputs(backend, [(None, hidden, {})], [layer])
    assert not layer._forward_pre_hooks


@torch.inference_mode()
def test_sequential_layer_artifact_matches_reconstructed_weights():
    torch.manual_seed(60)
    config = GPTQConfig(calibration_samples=4)
    layer = Layer()
    samples = [(torch.randn(1, 2 + i, 128).bfloat16(), {}) for i in range(4)]
    original = {name: layer.get_submodule(name).weight.clone() for name in PROJECTIONS}
    entry, outputs = quantize_layer(layer, samples, config, "cpu")
    assert set(entry) == set(PROJECTIONS) and len(outputs) == 4
    for name, item in entry.items():
        codes = unpack_signed(item["packed_codes"], 4, 128)
        expected = (codes.float() * item["scales"].repeat_interleave(128, 1)).bfloat16()
        assert torch.equal(layer.get_submodule(name).weight, expected)
        assert not torch.equal(original[name], expected)
        assert not layer.get_submodule(name)._forward_pre_hooks
    for (hidden, _), (result, arguments) in zip(samples, outputs):
        assert torch.equal(result, layer(hidden)[0]) and arguments == {}


def artifact_fixture(root):
    root.mkdir()
    files = {}
    for i in range(32):
        name = f"layer_{i:02d}.pt"
        (root / name).write_bytes(b"fixture")
        files[name] = digest_file(root / name)
    config = GPTQConfig(calibration_samples=4)
    value = dict(
        schema_version=1,
        algorithm="gptq",
        complete=True,
        layers=32,
        config=config.calibration_parameters(),
        files=files,
        numerical_contract=CONTRACT,
        calibration_identities=[
            dict(suite=s, split="train", episode_index=0, observation_sha256="a" * 64)
            for s in SUITES
        ],
    )
    (root / "manifest.json").write_text(json.dumps(value))
    return config, value


def test_manifest_checksums_coverage_and_provenance(tmp_path):
    root = tmp_path / "artifact"
    config, value = artifact_fixture(root)
    assert read_manifest(root, config) == value
    (root / "layer_00.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(root, config)


@pytest.mark.parametrize("mutation", ["unbalanced", "nontrain", "contract", "coverage", "config"])
def test_manifest_rejects_invalid_provenance(tmp_path, mutation):
    root = tmp_path / "artifact"
    config, value = artifact_fixture(root)
    if mutation == "unbalanced":
        value["calibration_identities"][0]["suite"] = "foreign"
    elif mutation == "nontrain":
        value["calibration_identities"][0]["split"] = "validation"
    elif mutation == "contract":
        value["numerical_contract"] = "round_to_nearest"
    elif mutation == "coverage":
        value["files"].pop("layer_31.pt")
    else:
        value["config"]["group_size"] = 64
    (root / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_manifest(root, config)


def test_native_precision_guard():
    layer = Layer().float()
    with pytest.raises(ValueError):
        validate_layer(layer)


def test_artifact_value_failure_does_not_partially_convert_model(tmp_path):
    pytest.importorskip("awq_inference_engine", reason="Optional packed runtime is not installed")
    from efficient_vla_benchmark.methods.quantization.gptq.method import GPTQ

    root = tmp_path / "atomic"
    config, manifest = artifact_fixture(root)
    layers = nn.ModuleList([Layer() for _ in range(32)])
    for i, layer in enumerate(layers):
        entry = {}
        for name in PROJECTIONS:
            weight = layer.get_submodule(name).weight
            entry[name] = dict(
                packed_codes=pack_signed(torch.zeros(128, 128, dtype=torch.int8), 4),
                scales=torch.full((128, 1), 0.01),
                shape=list(weight.shape),
                native_dtype=str(weight.dtype),
                bias=None,
            )
        if i == 31:
            entry[PROJECTIONS[-1]]["scales"][0] = float("nan")
        filename = f"layer_{i:02d}.pt"
        torch.save(entry, root / filename)
        manifest["files"][filename] = digest_file(root / filename)
    manifest.update(backend="fixture", checkpoint_sha256="b" * 64)
    (root / "manifest.json").write_text(json.dumps(manifest))
    backend = SimpleNamespace(
        method=None,
        device="cpu",
        metadata=lambda: dict(backend="fixture", checkpoint_sha256="b" * 64),
    )
    with pytest.raises(ValueError, match="scale"):
        GPTQ(backend, config, root, layers)
    assert all(
        isinstance(layer.get_submodule(name), nn.Linear) for layer in layers for name in PROJECTIONS
    )
