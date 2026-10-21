"""Final-only presets are the public execution surface, not a trial archive."""

from pathlib import Path

import yaml

from efficient_vla_benchmark.core.config import ExperimentConfig


def test_public_experiments_are_exactly_final_selection_and_published_controls():
    selection = yaml.safe_load(Path("configs/evaluation/primary_selection.yaml").read_text())
    chosen = selection["configurations"]
    paths = {Path(row["config"]) for row in chosen}
    controls = {
        Path(f"configs/experiments/cogact/{name}.yaml")
        for name in ("native_h16", "native_ddim4", "native_ddim5")
    }
    assert paths.isdisjoint(controls)
    assert paths | controls == set(Path("configs/experiments").glob("*/*.yaml"))
    identities = set()
    names = {
        "none": "baseline",
        "efficient_vla": "efficient_vla_token",
        "efficient_vla_layer_pruning": "efficient_vla_layer",
        "efficient_vla_diffusion_cache": "efficient_vla_cache",
    }
    for row in chosen:
        assert row["status"] == "primary"
        assert row["measured_preset"]
        path = Path(row["config"])
        config = ExperimentConfig.load(path)
        key = (config.backend, config.method)
        assert key not in identities
        identities.add(key)
        assert path.stem == names.get(config.method, config.method)
        assert path.parent.name == config.backend
        assert config.execution_horizon == 8
    for path in paths | controls:
        data = yaml.safe_load(path.read_text())
        parents = data.get("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        for parent in parents:
            target = (path.parent / parent).resolve()
            assert target.is_file()
            assert target.parent.name in {"backends", "methods"}


def test_reusable_method_defaults_are_all_used_by_final_presets():
    parents = set()
    for path in Path("configs/experiments").glob("*/*.yaml"):
        data = yaml.safe_load(path.read_text())
        values = data.get("extends", [])
        if isinstance(values, str):
            values = [values]
        parents.update((path.parent / name).resolve() for name in values)
    assert {p.resolve() for p in Path("configs/methods").glob("*.yaml")} <= parents
