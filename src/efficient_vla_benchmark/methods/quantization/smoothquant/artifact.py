"""Integrity and provenance checks for an immutable SmoothQuant calibration."""

import json

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.evaluation.libero import SUITES


def read_manifest(root, config):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("Missing SmoothQuant artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != 1
        or m.get("algorithm") != "smoothquant"
        or m.get("complete") is not True
    ):
        raise ValueError("Unsupported SmoothQuant artifact")
    if m.get("config") != config.calibration_parameters() or m.get("layers") != 32:
        raise ValueError("SmoothQuant configuration/coverage mismatch")
    expected = {f"layer_{i:02d}.pt" for i in range(32)}
    if set(m.get("files", {})) != expected:
        raise ValueError("Incomplete SmoothQuant layer artifacts")
    for name in expected:
        if not (root / name).is_file() or digest_file(root / name) != m["files"][name]:
            raise ValueError("SmoothQuant artifact checksum mismatch")
    identities = m.get("calibration_identities", [])
    if len(identities) != config.calibration_samples:
        raise ValueError("Incomplete calibration provenance")
    if any(
        sum(i.get("suite") == s for i in identities) != config.calibration_samples // 4
        for s in SUITES
    ):
        raise ValueError("Calibration is not balanced across four suites")
    if any(i.get("split") != "train" or not i.get("observation_sha256") for i in identities):
        raise ValueError("Calibration requires training observation provenance")
    if len({(i["suite"], i["episode_index"]) for i in identities}) != len(identities):
        raise ValueError("Calibration episodes must be distinct")
    return m
