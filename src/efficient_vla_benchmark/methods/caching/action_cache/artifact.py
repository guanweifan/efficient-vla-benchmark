"""Frozen successful-rollout bank, with no executable pickle payload."""

import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file


def read_manifest(path, config):
    if path is None:
        raise ValueError("ActionCache requires an explicit frozen artifact path")
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") != "action_cache_cogact_native_success_bank":
        raise ValueError("Wrong ActionCache artifact kind")
    schema = manifest.get("schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("Unsupported ActionCache bank schema")
    if schema == 2 and manifest.get("clean_action_source") != "native_final_ddim_sample":
        raise ValueError("Schema-2 clean actions lack native endpoint provenance")
    for key in ("key_dim", "projection_density", "projection_seed"):
        if manifest.get(key) != getattr(config, key):
            raise ValueError("ActionCache projection config differs from frozen bank")
    if digest_file(path / "bank.npz") != manifest.get("bank_sha256"):
        raise ValueError("ActionCache bank checksum mismatch")
    if manifest.get("episodes_disjoint_from_evaluation") is not True:
        raise ValueError("ActionCache population provenance is not evaluation-disjoint")
    if config.refinement_steps == 0 and (
        manifest.get("schema_version") != 2
        or manifest.get("clean_action_source") != "native_final_ddim_sample"
    ):
        raise ValueError("Zero-step ActionCache requires captured native clean actions (schema 2)")
    return manifest
