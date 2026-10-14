import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from efficient_vla_benchmark.methods.quantization.qvla.config import QVLAConfig
from efficient_vla_benchmark.methods.quantization.qvla.implementations.openvla_oft import (
    method,
)


def fixture(tmp_path, monkeypatch, invalid=False):
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.linear = nn.Linear(4, 2).bfloat16()
    original = model.language_model.linear
    original.cpu = Mock(wraps=original.cpu)
    before = original.weight.clone()
    name = "language_model.linear"
    (tmp_path / "gates.json").write_text(json.dumps({name: [4] if invalid else [4, 4]}))
    manifest = {
        "checkpoint_sha256": "fixture",
        "target_modules": [name],
        "inactive_target_modules": [],
        "stats": {"final_avg_bits": 4.0},
    }
    monkeypatch.setattr(method, "read_manifest", lambda root, config: manifest)

    class Prepared(nn.Module):
        def to(self, device):
            original.cpu.assert_called_once()
            return self

    monkeypatch.setattr(method, "MixedBitLinear", lambda module, gates: Prepared())
    backend = SimpleNamespace(
        model=model,
        method=None,
        device="cpu",
        metadata=lambda: {"precision": "bfloat16", "checkpoint_sha256": "fixture"},
    )
    return backend, original, before


def test_original_is_offloaded_before_replacement_transfer(tmp_path, monkeypatch):
    backend, original, before = fixture(tmp_path, monkeypatch)
    actual = method.QVLA(backend, QVLAConfig(), tmp_path)
    assert actual.count == 1 and backend.model.language_model.linear is not original
    torch.testing.assert_close(original.weight, before, atol=0, rtol=0)


def test_invalid_artifact_does_not_offload_or_replace(tmp_path, monkeypatch):
    backend, original, before = fixture(tmp_path, monkeypatch, invalid=True)
    with pytest.raises(ValueError):
        method.QVLA(backend, QVLAConfig(), tmp_path)
    original.cpu.assert_not_called()
    assert backend.model.language_model.linear is original
    torch.testing.assert_close(original.weight, before, atol=0, rtol=0)
