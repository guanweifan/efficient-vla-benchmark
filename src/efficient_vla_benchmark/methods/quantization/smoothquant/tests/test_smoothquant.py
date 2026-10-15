import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.evaluation.libero import SUITES
from efficient_vla_benchmark.methods.quantization.smoothquant.artifact import read_manifest
from efficient_vla_benchmark.methods.quantization.smoothquant.config import SmoothQuantConfig
from efficient_vla_benchmark.methods.quantization.smoothquant.mechanism import (
    GROUPS,
    PROJECTIONS,
    export_layer,
    migration_scale,
)
from efficient_vla_benchmark.methods.quantization.smoothquant.method import SmoothQuant
from efficient_vla_benchmark.model_ops.quantization.integer_linear import IntegerLinear


class Norm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.bfloat16))


def layer_fixture():
    layer = nn.Module()
    layer.input_layernorm = Norm(8)
    layer.post_attention_layernorm = Norm(8)
    layer.self_attn = nn.Module()
    layer.mlp = nn.Module()
    for name in PROJECTIONS:
        parent, attr = name.rsplit(".", 1)
        shape = (
            (8, 16)
            if name in ("mlp.gate_proj", "mlp.up_proj")
            else (16, 8)
            if name == "mlp.down_proj"
            else (8, 8)
        )
        setattr(
            layer.get_submodule(parent), attr, nn.Linear(*shape, bias=False, dtype=torch.bfloat16)
        )
    return layer


@pytest.mark.parametrize("alpha", [0, 0.5, 0.85, 1])
def test_shared_scale_and_exact_migration_equations(alpha):
    torch.manual_seed(7)
    weights = [torch.randn(11, 8), torch.randn(13, 8), torch.randn(9, 8)]
    amax = torch.rand(8) * 4 + 0.1
    scale = migration_scale(amax, weights, alpha)
    maximum = torch.cat(weights).abs().amax(0).clamp_min(1e-5)
    expected = (amax.pow(alpha) / maximum.pow(1 - alpha)).clamp_min(1e-5)
    torch.testing.assert_close(scale, expected, atol=0, rtol=0)
    x = torch.randn(3, 8, dtype=torch.float64)
    for weight in weights:
        torch.testing.assert_close(
            (x / scale.double()) @ (weight.double() * scale.double()).T,
            x @ weight.double().T,
            atol=1e-12,
            rtol=1e-12,
        )


@torch.inference_mode()
def test_export_preserves_native_and_uses_shared_group_scales():
    layer = layer_fixture()
    original = {name: value.clone() for name, value in layer.state_dict().items()}
    maxima = {group[0]: torch.linspace(0, 4, 8) for _, group in GROUPS}
    result = export_layer(layer, maxima, 0.85)
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
    assert set(result["projections"]) == set(PROJECTIONS)
    for norm, group in GROUPS:
        scale = migration_scale(
            maxima[group[0]], [layer.get_submodule(n).weight for n in group], 0.85
        )
        torch.testing.assert_close(
            result["norms"][norm], (layer.get_submodule(norm).weight.float() / scale).bfloat16()
        )
        for name in group:
            weight = (layer.get_submodule(name).weight.float() * scale).bfloat16().float()
            ws = weight.abs().amax(1).clamp_min(1e-5) / 127
            expected = (weight / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
            torch.testing.assert_close(
                result["projections"][name]["codes"], expected, atol=0, rtol=0
            )
            torch.testing.assert_close(result["projections"][name]["scales"], ws, atol=0, rtol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"alpha": True},
        {"alpha": float("nan")},
        {"alpha": 1.1},
        {"calibration_samples": 3},
        {"calibration_samples": True},
        {"calibration_seed": -1},
        {"bits": 4},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        SmoothQuantConfig.parse(settings)


@pytest.fixture
def artifact(tmp_path):
    config = SmoothQuantConfig.parse({"calibration_samples": 4})
    layers = nn.ModuleList([layer_fixture() for _ in range(32)])
    maxima = {group[0]: torch.ones(8) for _, group in GROUPS}
    files = {}
    for i, layer in enumerate(layers):
        name = f"layer_{i:02d}.pt"
        torch.save(export_layer(layer, maxima, config.alpha), tmp_path / name)
        files[name] = digest_file(tmp_path / name)
    manifest = dict(
        schema_version=1,
        algorithm="smoothquant",
        complete=True,
        config=config.calibration_parameters(),
        layers=32,
        files=files,
        backend="fixture",
        checkpoint_sha256="checkpoint",
        numerical_contract="fp32_statistics_native_folded_weights_tokenwise_int8_v1",
        calibration_identities=[
            dict(suite=s, split="train", episode_index=0, observation_sha256=s) for s in SUITES
        ],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    backend = SimpleNamespace(
        method=None,
        device="cpu",
        metadata=lambda: dict(backend="fixture", checkpoint_sha256="checkpoint"),
    )
    return tmp_path, config, layers, manifest, backend


def test_full_coverage_conversion_and_reload(artifact):
    root, config, layers, _, backend = artifact
    method = SmoothQuant(backend, config, root, layers)
    assert sum(isinstance(m, IntegerLinear) for layer in layers for m in layer.modules()) == 224
    assert method.metadata()["integer_mma"] and not method.reversible
    method.reset(None)
    method.close()
    method.close()
    with pytest.raises(ValueError, match="fresh native"):
        SmoothQuant(backend, config, root, layers)


def test_foreign_artifact_and_missing_data_rejected(artifact):
    root, config, layers, manifest, backend = artifact
    manifest["backend"] = "other"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint/backend"):
        SmoothQuant(backend, config, root, layers)
    assert isinstance(layers[0].self_attn.q_proj, nn.Linear)
    (root / "layer_31.pt").unlink()
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(root, config)


def test_last_layer_error_does_not_partially_convert_model(artifact):
    root, config, layers, manifest, backend = artifact
    entry = torch.load(root / "layer_31.pt", weights_only=True)
    entry["projections"]["self_attn.q_proj"]["native_dtype"] = "torch.float32"
    torch.save(entry, root / "layer_31.pt")
    manifest["files"]["layer_31.pt"] = digest_file(root / "layer_31.pt")
    (root / "manifest.json").write_text(json.dumps(manifest))
    before = layers[0].input_layernorm.weight.clone()
    with pytest.raises(ValueError, match="geometry/precision"):
        SmoothQuant(backend, config, root, layers)
    assert isinstance(layers[0].self_attn.q_proj, nn.Linear)
    torch.testing.assert_close(layers[0].input_layernorm.weight, before, atol=0, rtol=0)


@pytest.mark.parametrize("fault", ["duplicate", "evaluation", "unbalanced"])
def test_training_only_provenance(artifact, fault):
    root, config, _, manifest, _ = artifact
    if fault in ("duplicate", "unbalanced"):
        manifest["calibration_identities"][1]["suite"] = manifest["calibration_identities"][0][
            "suite"
        ]
    else:
        manifest["calibration_identities"][0]["split"] = "eval"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(root, config)
