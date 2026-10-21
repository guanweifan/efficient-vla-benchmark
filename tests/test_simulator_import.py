"""Simulator bootstrap must not depend on ownership of another user's log."""

import logging
from types import SimpleNamespace

import pytest

from efficient_vla_benchmark.evaluation.libero import runtime


@pytest.mark.parametrize("fail_import", [False, True])
def test_legacy_log_redirect_is_narrow_and_restored(tmp_path, monkeypatch, fail_import):
    original = logging.FileHandler
    environment = object()
    private = tmp_path / "private"
    unrelated = tmp_path / "other.log"

    def import_simulator(name):
        assert name == "libero.libero.envs"
        legacy = logging.FileHandler("/tmp/robosuite.log")
        other = logging.FileHandler(unrelated)
        try:
            assert legacy.baseFilename == str(private / "robosuite.log")
            assert other.baseFilename == str(unrelated)
            assert isinstance(legacy, original)
        finally:
            legacy.close()
            other.close()
        if fail_import:
            raise ImportError("simulator dependency missing")
        return SimpleNamespace(OffScreenRenderEnv=environment)

    monkeypatch.setattr(runtime.importlib, "import_module", import_simulator)
    if fail_import:
        with pytest.raises(ImportError, match="dependency missing"):
            runtime.load_environment(private)
    else:
        assert runtime.load_environment(private) is environment
    assert logging.FileHandler is original
    assert (private / "robosuite.log").is_file()
    assert unrelated.is_file()
