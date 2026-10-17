"""Strict dynamic-quantization artifact and checkpoint identities."""

import json

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash


def read_manifest(root, config, runtime_path=None):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("DyQ requires its independent-trajectory artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != (3 if config.runtime == "mixed_integer" else 2)
        or m.get("algorithm") != "dyq_vla"
        or m.get("complete") is not True
    ):
        raise ValueError("Handover/proxy or incomplete DyQ artifacts are not admitted")
    if m.get("config") != config.calibration_parameters():
        raise ValueError("DyQ artifact configuration mismatch")
    if m.get("files") != {"weights.pt": digest_file(root / "weights.pt")}:
        raise ValueError("DyQ artifact weights checksum mismatch")
    if m.get("population_kind") != "independent_native_quantization_population":
        raise ValueError("Missing independent DyQ trajectory provenance")
    if len(m.get("sample_identities", [])) < 4 or len(m.get("error_records", [])) != len(
        m["sample_identities"]
    ):
        raise ValueError("Incomplete DyQ threshold calibration")
    if stable_hash(m["sample_identities"]) != m["sample_identities_sha256"]:
        raise ValueError("DyQ calibration identity mismatch")
    if config.runtime == "mixed_integer":
        from efficient_vla_benchmark.model_ops.quantization.mixed_runtime import read_runtime

        if m.get("execution_runtime") != read_runtime(runtime_path):
            raise ValueError("DyQ thresholds require the exact calibrated integer runtime")
    return m
