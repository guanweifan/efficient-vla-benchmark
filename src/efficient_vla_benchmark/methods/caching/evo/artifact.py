"""Frozen searched layouts; runtime cannot optimize or use evaluation feedback."""

import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from .search import validate_layout


def load_artifact(root, config):
    if root is None:
        raise ValueError("EVO requires an independently searched, frozen schedule")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "evo"
        or manifest.get("complete") is not True
        or manifest.get("backend") != "cogact"
        or manifest.get("sha256")
        != stable_hash({k: v for k, v in manifest.items() if k != "sha256"})
        or digest_file(root / "search.json") != manifest.get("search_sha256")
        or manifest.get("uses_dev_or_final_feedback") is not False
    ):
        raise ValueError("Incomplete, modified or nonindependent EVO artifact")
    try:
        candidate = manifest["layouts"][str(config.refresh_budget)]
    except KeyError as error:
        raise ValueError("Requested global budget has no frozen searched layout") from error
    validate_layout(
        candidate["layout"], manifest["num_blocks"], manifest["num_steps"], config.refresh_budget
    )
    return manifest
