"""Content-addressed exported weights, calibration and compiled-engine identity."""

import hashlib
import json

from efficient_vla_benchmark.core.provenance import digest_file

from .config import ActQuantConfig
from .vendor import REVISION, reference_source

WEIGHTS = (
    "dinov2_f32_patch.gguf",
    "siglip_f32_patch.gguf",
    "proj.gguf",
    "llm.gguf",
    "action_head.gguf",
    "proprio_projector.gguf",
)


def identity(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def read_manifest(root, config, runtime_path):
    reference_source(runtime_path)
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("ActQuant needs a complete exported artifact")
    manifest = json.loads((root / "manifest.json").read_text())
    quantized = isinstance(config, ActQuantConfig)
    scope = "hsic_amf_quantized" if quantized else "floating_runtime_control"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != scope
        or manifest.get("reference_revision") != REVISION
        or manifest.get("complete") is not True
    ):
        raise ValueError("ActQuant artifact identity or completion mismatch")
    if set(manifest.get("files", {})) != set(WEIGHTS):
        raise ValueError("ActQuant artifact has incomplete or unexpected weight files")
    for name, digest in manifest["files"].items():
        if not (root / name).is_file() or digest_file(root / name) != digest:
            raise ValueError("ActQuant exported weight checksum mismatch")
    binary = runtime_path / "build/bin/openvla_oft.so"
    if not binary.is_file() or digest_file(binary) != manifest.get("runtime_binary_sha256"):
        raise ValueError("ActQuant compiled runtime differs from calibrated artifact")
    if quantized:
        from dataclasses import asdict

        if manifest.get("config") != asdict(config):
            raise ValueError("ActQuant configuration differs from allocated artifact")
        if not manifest.get("calibration_sha256") or manifest.get("quantized_tensors") != 224:
            raise ValueError("ActQuant requires complete HSIC and AMF provenance")
        if digest_file(root / "allocation.json") != manifest.get("allocation_sha256"):
            raise ValueError("ActQuant allocation checksum mismatch")
    return manifest
