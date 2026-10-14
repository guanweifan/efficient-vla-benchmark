"""Execution-runtime lifecycle without allocating a checkpoint or GPU."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend


def fixture(monkeypatch):
    b = OpenVLAOFTBackend.__new__(OpenVLAOFTBackend)
    b.initialize_method_state()
    b.context = b.execution_runtime = None
    b.device = torch.device("cuda:0")
    b.model, b.action_head, b.proprio_projector = Mock(), Mock(), Mock()
    b.processor = Mock()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    r = SimpleNamespace(
        **{name: Mock() for name in ("prepare", "forward", "reset", "close", "metadata")},
        monitor=Mock(),
    )
    return b, r


def test_external_runtime_is_fresh_only_and_does_not_retain_gpu_weights(monkeypatch):
    b, r = fixture(monkeypatch)
    b.install_execution_runtime(r)
    b.model.cpu.assert_called_once()
    b.action_head.cpu.assert_called_once()
    b.proprio_projector.cpu.assert_called_once()
    assert b.runtime is r.monitor
    observation, prepared = object(), object()
    r.prepare.return_value = prepared
    assert b.prepare(observation) is prepared
    b.forward(prepared)
    r.forward.assert_called_once_with(prepared)
    with pytest.raises(ValueError, match="fresh"):
        b.install_execution_runtime(r)
    b.close()
    b.close()
    r.close.assert_called_once()
    assert b.model is None and b._closed


def test_runtime_failure_still_releases_native_resources(monkeypatch):
    b, r = fixture(monkeypatch)
    b.install_execution_runtime(r)
    r.close.side_effect = RuntimeError("engine failure")
    with pytest.raises(RuntimeError, match="engine failure"):
        b.close()
    assert b._closed and b.model is None and b.execution_runtime is None
    b.close()


def test_invalid_contract_does_not_move_native_modules(monkeypatch):
    b, _ = fixture(monkeypatch)
    with pytest.raises(ValueError, match="contract"):
        b.install_execution_runtime(object())
    b.model.cpu.assert_not_called()
