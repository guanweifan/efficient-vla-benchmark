"""Change only the channel budget using frozen, training-only sensitivity scores."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from .allocation import greedy_allocate
from .artifact import read_manifest
from .config import QVLAConfig


def reallocate(root, output, target_avg_bits, candidate_bits=None):
    import torch

    previous = json.loads((root / "manifest.json").read_text())
    config = QVLAConfig.parse(previous["config"])
    previous = read_manifest(root, config)
    updated = replace(
        config,
        target_avg_bits=target_avg_bits,
        candidate_bits=config.candidate_bits if candidate_bits is None else tuple(candidate_bits),
    )
    QVLAConfig.parse(updated.calibration_parameters())
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a new artifact outside the calibration directory")
    proxies = torch.load(root / "sensitivity.pt", map_location="cpu", weights_only=True)
    if any(not set(updated.candidate_bits).issubset(scores) for scores in proxies.values()):
        raise ValueError("Requested candidate has no frozen sensitivity evidence")
    gates, stats = greedy_allocate(proxies, updated.candidate_bits, updated.target_avg_bits)
    if set(proxies) != set(previous["target_modules"]):
        raise ValueError("Sensitivity target coverage mismatch")
    output.mkdir(parents=True)
    shutil.copyfile(root / "sensitivity.pt", output / "sensitivity.pt")
    (output / "gates.json").write_text(json.dumps(gates) + "\n")
    manifest = {
        **previous,
        "config": updated.calibration_parameters(),
        "stats": stats,
        "files": {n: digest_file(output / n) for n in ("sensitivity.pt", "gates.json")},
        "calibration_source": previous.get("calibration_source", previous["source"]),
        "source": source_metadata(),
        "parent_manifest_sha256": digest_file(root / "manifest.json"),
        "additional_calibration_observations": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-avg-bits", type=float, required=True)
    parser.add_argument("--candidate-bits", type=int, nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            reallocate(args.artifact_path, args.output, args.target_avg_bits, args.candidate_bits)[
                "stats"
            ]
        )
    )


if __name__ == "__main__":
    main()
