"""Completed official-cost schema; legacy allocations are not formal artifacts."""

import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash


def read_manifest(root, config):
    if root is None:
        raise ValueError("QVLA requires a calibration artifact")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 5
        or manifest.get("algorithm") != "qvla"
        or manifest.get("complete") is not True
    ):
        raise ValueError("QVLA requires schema 5 literal calibration; legacy gates are unsupported")
    if manifest.get("backend") != "openvla_oft" or manifest.get("precision") != "bfloat16":
        raise ValueError("Unsupported QVLA artifact backend/precision")
    if manifest.get("config") != config.calibration_parameters():
        raise ValueError("QVLA calibration configuration mismatch")
    if (
        manifest.get("reference_revision") != "26cc4821a3be4c003d09d3c7997b38db2a347982"
        or manifest.get("allocation") != "official_absolute_cost_per_saved_bit"
        or manifest.get("sensitivity") != "official_literal_recurrence_contiguous_diag_division"
    ):
        raise ValueError("QVLA calibration mechanism mismatch")
    if set(manifest.get("files", {})) != {"sensitivity.pt", "gates.json"}:
        raise ValueError("Unexpected QVLA artifact file set")
    for name, checksum in manifest["files"].items():
        if digest_file(root / name) != checksum:
            raise ValueError("QVLA artifact checksum mismatch")
    return manifest


def validate_gates(manifest, gates, modules, config):
    active = manifest["target_modules"]
    inactive = manifest["inactive_target_modules"]
    if (
        len(active) != len(set(active))
        or len(inactive) != len(set(inactive))
        or set(active) & set(inactive)
        or set(active) | set(inactive) != set(modules)
        or set(gates) != set(active)
    ):
        raise ValueError("QVLA target coverage mismatch")
    total = count = 0
    for name, row in gates.items():
        if (
            not isinstance(row, list)
            or len(row) != modules[name].weight.shape[0]
            or any(type(b) is not int or b not in config.candidate_bits for b in row)
        ):
            raise ValueError("Invalid QVLA output-channel gates")
        total += sum(row)
        count += len(row)
    if not count or total / count > config.target_avg_bits + 1e-12:
        raise ValueError("QVLA allocation exceeds its channel budget")
    if abs(total / count - manifest["stats"]["final_avg_bits"]) > 1e-12:
        raise ValueError("QVLA gate statistics mismatch")
    return stable_hash(manifest)
