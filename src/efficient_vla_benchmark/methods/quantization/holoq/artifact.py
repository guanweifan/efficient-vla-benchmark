"""Strict identity for the official-code HoloQ port; handover artifacts are not interchangeable."""

import json

from efficient_vla_benchmark.core.provenance import digest_file


def read_manifest(root, config):
    if (
        root is None
        or not (root / "manifest.json").is_file()
        or not (root / "weights.pt").is_file()
    ):
        raise ValueError("HoloQ requires its calibrated artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != 1
        or m.get("algorithm") != "holoq"
        or m.get("complete") is not True
    ):
        raise ValueError("Unsupported/incomplete HoloQ artifact")
    if m.get("variant") != "official_code_composite_gptq_perstep_cogact":
        raise ValueError("HoloQ core variant mismatch")
    if m.get("config") != config.calibration_parameters():
        raise ValueError("HoloQ calibration configuration mismatch")
    if m.get("files") != {"weights.pt": digest_file(root / "weights.pt")}:
        raise ValueError("HoloQ artifact checksum mismatch")
    if len(m.get("calibration_identities", [])) != config.calibration_samples:
        raise ValueError("Incomplete HoloQ calibration provenance")
    return m
