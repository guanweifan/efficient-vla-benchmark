import json

from efficient_vla_benchmark.core.provenance import digest_file


def read_manifest(root, config):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("DA-MPA needs a completed calibration artifact")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema_version") != 1
        or m.get("algorithm") != "da_mpa"
        or m.get("complete") is not True
        or m.get("variant") != "DA_MPA_only_no_CSRC"
    ):
        raise ValueError(
            "Foreign/incomplete DA-MPA artifact; full DA-PTQ proxies are not compatible"
        )
    if m.get("config") != config.calibration_parameters() or m.get("files") != {
        "weights.pt": digest_file(root / "weights.pt")
    }:
        raise ValueError("DA-MPA configuration or checksum mismatch")
    if (
        len(m.get("calibration_identities", [])) != config.calibration_samples
        or len(m.get("probe_indices", [])) != config.probe_steps
    ):
        raise ValueError("Incomplete DA-MPA calibration provenance")
    return m
