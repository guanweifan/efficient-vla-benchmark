"""Portable, checksummed packed decoder artifacts; base checkpoints stay read-only."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from .runtime import KernelPrecisionLinear
from .vendor import reference_module

LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def replace_linear(layer, name, replacement):
    parent, child = name.rsplit(".", 1)
    setattr(layer.get_submodule(parent), child, replacement)


def read_manifest(root: Path, config):
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("AWQ needs a completed calibration artifact with manifest.json")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "awq"
        or manifest.get("complete") is not True
    ):
        raise ValueError("Unsupported or incomplete AWQ artifact")
    if manifest.get("config") != config.calibration_parameters():
        raise ValueError("AWQ artifact settings do not match the supplied experiment configuration")
    if manifest.get("scope") != "language_decoder_block_linears":
        raise ValueError("Unexpected AWQ quantization scope")
    components = manifest.get("components", [])
    if len(components) != 32:
        raise ValueError("AWQ artifact must contain all 32 decoder layers")
    for index, component in enumerate(components):
        if component.get("file") != f"layer_{index:02d}.safetensors":
            raise ValueError("Unexpected AWQ component filename")
    return manifest


def load_artifact(backend, root: Path, config):
    manifest = read_manifest(root, config)
    if manifest.get("checkpoint_sha256") != backend._model_manifest["sha256"]:
        raise ValueError("AWQ artifact was calibrated for a different base checkpoint")
    if manifest.get("precision") != backend.precision:
        raise ValueError("AWQ artifact activation precision differs from the runtime")
    WQLinear = reference_module("quantize.qmodule").WQLinear
    layers = backend.model.language_model.model.layers
    # Validate every component before mutating any model layer.
    for component in manifest["components"]:
        path = root / component["file"]
        if not path.is_file() or digest_file(path) != component["sha256"]:
            raise ValueError("AWQ component is missing or fails its content checksum")
    for layer, component in zip(layers, manifest["components"]):
        for name in LINEARS:
            original = layer.get_submodule(name)
            packed = WQLinear.from_linear(original, config.bits, config.group_size, init_only=True)
            replace_linear(layer, name, packed)
        layer.load_state_dict(load_file(str(root / component["file"])), strict=True)
        kernel_dtype = (
            backend.dtype
            if config.kernel_precision == "auto"
            else getattr(torch, config.kernel_precision)
        )
        if kernel_dtype != backend.dtype:
            for name in LINEARS:
                replace_linear(
                    layer, name, KernelPrecisionLinear(layer.get_submodule(name), kernel_dtype)
                )
    return stable_hash(manifest)
