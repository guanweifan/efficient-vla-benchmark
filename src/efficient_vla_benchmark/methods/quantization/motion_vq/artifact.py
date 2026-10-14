import json

from efficient_vla_benchmark.core.provenance import digest_file


def read_manifest(root, config):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("Missing MotionVQ artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != 1
        or m.get("algorithm") != "motion_vq"
        or m.get("complete") is not True
        or m.get("variant") != "official_code_LLM_only_MotionVQ"
    ):
        raise ValueError("Foreign or incomplete MotionVQ artifact")
    if m.get("config") != config.calibration_parameters() or m.get("files") != {
        "weights.pt": digest_file(root / "weights.pt")
    }:
        raise ValueError("MotionVQ identity/checksum mismatch")
    if len(m.get("calibration_identities", [])) != config.calibration_samples:
        raise ValueError("Incomplete calibration provenance")
    return m
