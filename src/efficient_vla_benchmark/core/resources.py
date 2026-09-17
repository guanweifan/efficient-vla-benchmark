"""Resolve project-owned sources without hard-coded machine locations."""

from pathlib import Path


def repository_root() -> Path | None:
    for root in Path(__file__).resolve().parents:
        manifest = root / "pyproject.toml"
        if manifest.is_file() and 'name = "efficient-vla-benchmark"' in manifest.read_text():
            return root
    return None


def project_source(relative: str) -> Path:
    root = repository_root()
    if root is None:
        raise RuntimeError(
            "This source integration needs an editable checkout with pinned submodules"
        )
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError("Initialize the selected project source dependency first")
    return path
