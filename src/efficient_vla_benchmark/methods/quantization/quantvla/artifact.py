"""Strict schema for new, source-audited calibration; old handover artifacts fail closed."""

import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

REVISION = "9e435e56aac4e743526557e5d9ba6e4fddd6ccca"


def read_manifest(root, config):
    if root is None:
        raise ValueError("QuantVLA needs a frozen calibration artifact")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 3
        or manifest.get("algorithm") != "quantvla"
        or manifest.get("complete") is not True
    ):
        raise ValueError(
            "QuantVLA requires source-audited schema 3; old handover gates are diagnostic only"
        )
    if manifest.get("backend") != "cogact" or manifest.get("precision") != "bfloat16":
        raise ValueError("Unsupported QuantVLA artifact backend/precision")
    if (
        manifest.get("config") != config.calibration_parameters()
        or manifest.get("reference_revision") != REVISION
        or manifest.get("execution") != config.execution()
    ):
        raise ValueError("QuantVLA artifact configuration/source mismatch")
    files = manifest.get("files", {})
    if set(files) != {"weights.pt", "attention.json"}:
        raise ValueError("Unexpected QuantVLA artifact files")
    for filename, expected in files.items():
        if digest_file(root / filename) != expected:
            raise ValueError("QuantVLA artifact checksum mismatch")
    return manifest


def artifact_identity(manifest):
    return stable_hash(manifest)
