"""Reject incomplete, stale or foreign calibration artifacts."""

import json

from efficient_vla_benchmark.core.provenance import digest_file


def read_manifest(root, config):
    if (
        root is None
        or not (root / "manifest.json").is_file()
        or not (root / "weights.pt").is_file()
    ):
        raise ValueError("Missing EaqVLA artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != 1
        or m.get("algorithm") != "eaqvla"
        or m.get("complete") is not True
    ):
        raise ValueError("Unsupported EaqVLA artifact")
    if m.get("config") != config.calibration_parameters() or m.get("files") != {
        "weights.pt": digest_file(root / "weights.pt")
    }:
        raise ValueError("EaqVLA configuration/checksum mismatch")
    if len(m.get("calibration_identities", [])) != config.calibration_samples:
        raise ValueError("Incomplete calibration provenance")
    return m
