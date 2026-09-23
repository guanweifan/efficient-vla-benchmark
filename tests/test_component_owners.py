from pathlib import Path

import pytest
import yaml

from efficient_vla_benchmark.core.registry import builtin_method_specs


def owner(tmp_path: Path):
    paper = tmp_path / "example"
    paper.mkdir()
    (paper / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "example",
                "category": "token_reduction",
                "components_only": True,
            }
        )
    )
    (paper / "components").mkdir()
    return paper


def test_component_only_owner_does_not_advertise_full_method(tmp_path):
    paper = owner(tmp_path)
    (paper / "components" / "primary.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "example_primary",
                "category": "token_reduction",
                "config": "example.config:Config",
                "implementations": {"fixture": {"variants": ["test"], "factory": "fixture:build"}},
            }
        )
    )
    assert [spec.name for spec in builtin_method_specs(paper, "token_reduction")] == [
        "example_primary"
    ]


def test_empty_component_owner_is_not_a_placeholder(tmp_path):
    with pytest.raises(ValueError, match="runnable core"):
        builtin_method_specs(owner(tmp_path), "token_reduction")
