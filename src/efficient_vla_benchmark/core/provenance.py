"""Portable identities, never machine-local resource locations."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from efficient_vla_benchmark.core.resources import repository_root


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path, names) -> dict:
    """Hash a caller-declared asset set; component names are backend-owned."""
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": digest_file(root / name)}
        for name in sorted(names)
        if (root / name).is_file()
    }
    return {"files": files, "sha256": stable_hash(files)}


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_metadata() -> dict:
    root = repository_root()

    def git(*args):
        if root is None:
            return ""
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    source_files = {}
    if root is not None:
        for name in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines():
            path = root / name
            if (
                path.is_file()
                and path.suffix in {".py", ".yaml", ".toml", ".txt"}
                and path.parts
                and "upstream" not in Path(name).parts
                and not name.startswith(("third_party/", "docs/", "tests/", "templates/"))
            ):
                source_files[name] = digest_file(path)
    else:
        package = Path(__file__).resolve().parents[1]
        source_files = {
            str(path.relative_to(package)): digest_file(path)
            for path in sorted(package.rglob("*.py"))
        }
    fingerprint = stable_hash(source_files)
    runtime_files = {
        name: value
        for name, value in source_files.items()
        if name.startswith(("src/", "backends/"))
    }

    submodules = {}
    for line in git("submodule", "status").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            submodules[fields[1]] = fields[0].lstrip("+-U")
    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "robosuite",
        "mujoco",
        "flash-attn",
        "awq-inference-engine",
        "tensorflow",
        "tensorflow-datasets",
        "timm",
        "tokenizers",
        "diffusers",
        "efficient-vla-benchmark",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    for entry in importlib.metadata.entry_points(group="efficient_vla_benchmark.backends"):
        if entry.dist is not None:
            packages[entry.dist.name] = entry.dist.version
    extensions = {}
    for name in ("flash-attn", "awq-inference-engine"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        extensions[name] = {
            Path(str(file)).name: digest_file(Path(distribution.locate_file(file)))
            for file in distribution.files or []
            if str(file).endswith(".so")
        }
    return {
        "commit": git("rev-parse", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": fingerprint,
        "runtime_source_sha256": stable_hash(runtime_files),
        "submodules": submodules,
        "packages": packages,
        "cuda_extension_sha256": extensions,
    }


def driver_version() -> str | None:
    """Driver provenance without exporting device UUIDs, machine names or addresses."""
    try:
        versions = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    distinct = sorted(set(version.strip() for version in versions if version.strip()))
    return ",".join(distinct) or None
