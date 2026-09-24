import importlib.util
import logging
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/gpu_worker.py"
    spec = importlib.util.spec_from_file_location("gpu_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mask", [None, "", "0,1"])
def test_requires_one_visible_gpu(launcher, mask):
    with pytest.raises(ValueError, match="exactly one"):
        launcher.select_gpu(mask)


def test_resolves_numeric_selection_to_uuid(launcher, monkeypatch):
    calls = []

    def output(command, **kwargs):
        calls.append(command)
        return "GPU-fixture\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", output)
    assert launcher.select_gpu("3") == "GPU-fixture"
    assert "--id=3" in calls[0]


@pytest.mark.parametrize("output", ["", "GPU-a\nGPU-b\n", "unknown\n"])
def test_rejects_ambiguous_device_selection(launcher, monkeypatch, output):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: output)
    with pytest.raises(ValueError, match="exactly one UUID"):
        launcher.select_gpu("0")


def test_simulator_log_is_private_and_handler_is_restored(launcher, tmp_path):
    original = logging.FileHandler
    with launcher.private_simulator_log(tmp_path / "private"):
        handler = logging.FileHandler("robosuite.log")
        assert Path(handler.baseFilename) == tmp_path / "private/robosuite.log"
        handler.close()
        other = logging.FileHandler(tmp_path / "other.log")
        assert Path(other.baseFilename) == tmp_path / "other.log"
        other.close()
    assert logging.FileHandler is original


def test_simulator_log_handler_restores_on_failure(launcher, tmp_path):
    original = logging.FileHandler
    with pytest.raises(RuntimeError):
        with launcher.private_simulator_log(tmp_path):
            raise RuntimeError("import failed")
    assert logging.FileHandler is original


def test_probe_does_not_import_simulator(launcher, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(launcher, "select_gpu", lambda mask: "GPU-fixture")
    monkeypatch.setattr(launcher, "egl_index", lambda: 7)
    monkeypatch.setattr(launcher.sys, "argv", ["launcher", "--probe"])
    launcher.main()
    assert '"egl_index": 7' in capsys.readouterr().out
