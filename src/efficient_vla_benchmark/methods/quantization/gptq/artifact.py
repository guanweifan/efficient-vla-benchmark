"""Immutable, checkpoint-matched GPTQ calibration provenance."""

import json

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.evaluation.libero import SUITES

from .mechanism import CONTRACT


def read_manifest(root, config):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("Missing GPTQ calibration artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != 1
        or m.get("algorithm") != "gptq"
        or m.get("complete") is not True
        or m.get("numerical_contract") != CONTRACT
    ):
        raise ValueError("Unsupported GPTQ artifact")
    if m.get("config") != config.calibration_parameters() or m.get("layers") != 32:
        raise ValueError("GPTQ configuration/coverage mismatch")
    expected = {f"layer_{i:02d}.pt" for i in range(32)}
    if set(m.get("files", {})) != expected:
        raise ValueError("Incomplete GPTQ target coverage")
    for name in expected:
        if not (root / name).is_file() or digest_file(root / name) != m["files"][name]:
            raise ValueError("GPTQ artifact checksum mismatch")
    identities = m.get("calibration_identities", [])
    if len(identities) != config.calibration_samples or any(
        sum(i.get("suite") == suite for i in identities) != config.calibration_samples // 4
        for suite in SUITES
    ):
        raise ValueError("Incomplete or unbalanced GPTQ calibration provenance")
    if any(i.get("split") != "train" or not i.get("observation_sha256") for i in identities):
        raise ValueError("GPTQ calibration must use training observations")
    if len({(i["suite"], i["episode_index"]) for i in identities}) != len(identities):
        raise ValueError("GPTQ calibration episodes must be distinct")
    return m
