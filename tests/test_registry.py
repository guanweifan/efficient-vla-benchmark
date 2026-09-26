import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from efficient_vla_benchmark.core.backend import BackendSpec
from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.registry import (
    backend_entries,
    backend_spec,
    builtin_method_specs,
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


def test_all_presets_for_installed_backends_pass_public_selection():
    from efficient_vla_benchmark.core.registry import validate_selection

    installed = backend_entries()
    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    checked = 0
    for path in sorted(root.glob("*/*.yaml")):
        config = ExperimentConfig.load(path)
        if config.backend in installed:
            validate_selection(config)
            checked += 1
    if not checked:
        pytest.skip("No optional backend plugin installed")


def test_components_keep_one_paper_owner_and_explicit_identity(tmp_path):
    import yaml

    paper = tmp_path / "example"
    paper.mkdir()
    data = {
        "schema_version": 1,
        "name": "example",
        "category": "token_reduction",
        "config": "example:Config",
        "implementations": {"example": {"variants": ["default"], "factory": "x:y"}},
    }
    (paper / "metadata.yaml").write_text(yaml.safe_dump(data))
    components = paper / "components"
    components.mkdir()
    data.update(name="example_feature_cache", category="caching")
    component = components / "feature_cache.yaml"
    component.write_text(yaml.safe_dump(data))
    specs = builtin_method_specs(paper, "token_reduction")
    assert [(s.name, s.category) for s in specs] == [
        ("example", "token_reduction"),
        ("example_feature_cache", "caching"),
    ]
    data["name"] = "another_paper"
    component.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="owning paper"):
        builtin_method_specs(paper, "token_reduction")


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


@pytest.mark.parametrize("name", ["vla_pruner", "team_vla", "efficient_vla", "vla_iap"])
def test_unresolved_cogact_core_does_not_silently_use_oft_or_cache_component(name):
    spec = BackendSpec("cogact", "x:y", "x:z", ("libero_joint",), frozenset())
    with pytest.raises(ValueError, match="Unsupported"):
        select_method(ExperimentConfig("cogact", method=name, model_variant="libero_joint"), spec)


def test_vote_without_native_overlap_is_not_registered_on_oft():
    spec = BackendSpec("openvla_oft", "x:y", "x:z", ("libero_joint",), frozenset())
    with pytest.raises(ValueError, match="Unsupported"):
        select_method(
            ExperimentConfig("openvla_oft", method="vote_ensemble", model_variant="libero_joint"),
            spec,
        )


@pytest.mark.parametrize(
    "implementation",
    [
        {"variants": "libero_joint", "factory": "x:y"},
        {"variants": ["default", "default"], "factory": "x:y"},
        {"variants": ["default"], "factory": 42},
        {"variants": ["default"], "factory": "x:y", "requires": "llama_decoder"},
    ],
)
def test_compatibility_metadata_cannot_use_ambiguous_strings(implementation):
    with pytest.raises(ValueError):
        parse_method(
            {
                "schema_version": 1,
                "name": "example",
                "category": "caching",
                "config": "example:Config",
                "implementations": {"example": implementation},
            }
        )
