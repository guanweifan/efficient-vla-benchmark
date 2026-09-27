"""Pin checked external resources without importing a policy or tensor runtime."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .corpus import digest


def verified_assets(directory, repository, revision):
    directory = Path(directory)
    manifest = json.loads((directory / "download_manifest.json").read_text())
    if manifest["repository"] != repository or manifest["revision"] != revision:
        raise ValueError("Codec assets do not match the audited release")
    for row in manifest["files"]:
        path = directory / row["name"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Asset path escapes the resource directory")
        if digest(path) != row["sha256"]:
            raise ValueError("Changed codec asset")
    return manifest


def load_module(path, name):
    path = Path(path)
    key = name + "_" + digest(path)[:16]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key]


def factory(specification):
    module, name = specification.split(":", 1)
    if not module.startswith("efficient_vla_benchmark.action_codecs.implementations."):
        raise ValueError("Expected an explicit action-codec implementation factory")
    return getattr(importlib.import_module(module), name)
