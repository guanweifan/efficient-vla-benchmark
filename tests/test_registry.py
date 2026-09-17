import subprocess
import sys
from types import SimpleNamespace

import pytest

from efficient_vla_benchmark.core.backend import BackendSpec
from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.registry import (
    backend_entries,
    backend_spec,
    method_catalog,
    parse_method,
    select_method,
)


def test_discovery_does_not_import_model_runtimes():
    code = (
        "import sys; from efficient_vla_benchmark.core.registry import "
        "backend_entries, method_catalog; backend_entries(); method_catalog(); "
        "assert not any(n in sys.modules for n in "
        "('torch','transformers','tensorflow','prismatic','libero','starVLA'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_uninstalled_backend_is_explicit():
    with pytest.raises(ValueError, match="not installed"):
        backend_spec("nonexistent_backend")


def test_duplicate_plugins_are_rejected(monkeypatch):
    entries = [SimpleNamespace(name="example"), SimpleNamespace(name="example")]
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **kwargs: entries)
    with pytest.raises(ValueError, match="Duplicate"):
        backend_entries()


def test_backend_api_version_is_checked(monkeypatch):
    spec = BackendSpec("example", "x:y", "x:z", ("default",), frozenset(), api_version=999)
    entry = SimpleNamespace(name="example", load=lambda: lambda: spec)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **kwargs: [entry])
    with pytest.raises(ValueError, match="incompatible"):
        backend_spec("example")


def test_unknown_or_incompatible_method_fails_before_model_loading():
    spec = BackendSpec("example", "x:y", "x:z", ("default",), frozenset())
    with pytest.raises(ValueError, match="not been integrated"):
        select_method(ExperimentConfig("example", method="unknown_method"), spec)


def test_all_builtin_metadata_is_valid():
    for name, spec in method_catalog().items():
        assert name == spec.name
        assert spec.implementations


def test_empty_metadata_is_not_a_fake_implementation():
    with pytest.raises(ValueError, match="at least one"):
        parse_method(
            {
                "schema_version": 1,
                "name": "example",
                "category": "caching",
                "config": "example:Config",
                "implementations": {},
            }
        )
