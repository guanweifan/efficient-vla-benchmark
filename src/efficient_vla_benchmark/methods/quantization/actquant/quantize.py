"""Allocate and quantize a validated floating export with the pinned author tools.

Run as a module. All external locations are explicit CLI inputs; this tool never
changes an original checkpoint, invokes training, or creates an evaluation result.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from .artifact import WEIGHTS, identity, read_manifest
from .config import ActQuantConfig, ActQuantFloatingConfig
from .vendor import REVISION, gguf_module, reference_source


def quantize(config, calibration, floating, runtime_path, output):
    source = reference_source(runtime_path)
    base = read_manifest(floating, ActQuantFloatingConfig(), runtime_path)
    manifest_path = calibration / "calibration_manifest.json"
    calibration_manifest = json.loads(manifest_path.read_text())
    if (
        calibration_manifest.get("complete") is not True
        or calibration_manifest.get("algorithm") != "actquant_calibration"
        or calibration_manifest.get("reference_revision") != REVISION
        or calibration_manifest.get("checkpoint_sha256") != base["checkpoint_sha256"]
        or calibration_manifest.get("config") != config.calibration_parameters()
        or calibration_manifest.get("optimizer_updates") != 0
    ):
        raise ValueError("ActQuant calibration identity mismatch")
    if set(calibration_manifest.get("files", {})) != {"hsic.json", "amf.gguf"}:
        raise ValueError("Incomplete ActQuant HSIC/AMF calibration")
    for name, digest in calibration_manifest["files"].items():
        if digest_file(calibration / name) != digest:
            raise ValueError("ActQuant calibration checksum mismatch")
    if output.exists():
        raise ValueError("Refusing to overwrite a derived artifact")
    resolved = output.resolve()
    for protected in (calibration, floating, runtime_path):
        if resolved == protected.resolve() or resolved.is_relative_to(protected.resolve()):
            raise ValueError("Derived weights must be separate from their input assets")
    output.mkdir(parents=True)
    allocation_path = output / "allocation.json"
    with (output / "allocation.log").open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                str(source / "tools/hsic/allocate.py"),
                "--scores",
                str(calibration / "hsic.json"),
                "--score-key",
                "F_out",
                "--target-bpw",
                str(config.target_bpw),
                "--base-type",
                config.base_type,
                "--max-type",
                config.max_type,
                "--n-layers",
                "32",
                "--output",
                str(allocation_path),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    allocation = json.loads(allocation_path.read_text())
    assignments = allocation["assignments"]
    expected = {
        f"blk.{i}.{p}.weight"
        for i in range(32)
        for p in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
    }
    if set(assignments) != expected or allocation["achieved_bpw"] > config.target_bpw:
        raise ValueError("Author allocation target coverage/budget mismatch")
    scores = json.loads((calibration / "hsic.json").read_text())["tensors"]
    if not any(v["F_out"] > 0 for v in scores.values()):
        raise ValueError("All HSIC scores are nonpositive; allocation would ignore sensitivity")
    quantizer = runtime_path / "build/bin/llama-quantize"
    command = [
        str(quantizer),
        "--imatrix",
        str(calibration / "amf.gguf"),
        "--token-embedding-type",
        "bf16",
        "--output-tensor-type",
        "bf16",
    ]
    # Set every projection explicitly, preventing unrelated default quantizer heuristics.
    for name, kind in sorted(assignments.items()):
        command.extend(["--tensor-type", f"^{re.escape(name)}$={kind}"])
    command.extend([str(floating / "llm.gguf"), str(output / "llm.gguf"), config.base_type, "4"])
    with (output / "quantizer.log").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    reader = gguf_module(runtime_path).GGUFReader(str(output / "llm.gguf"))
    tensor_types = {t.name: t.tensor_type.name for t in reader.tensors}
    if any(tensor_types.get(n) != kind for n, kind in assignments.items()):
        raise ValueError("Packed GGUF types differ from the HSIC allocation")
    if any(tensor_types.get(n) != "BF16" for n in ("token_embd.weight", "output.weight")):
        raise ValueError("Non-target embedding/output precision was changed")
    del reader
    for name in WEIGHTS:
        if name != "llm.gguf":
            os.link(floating / name, output / name)
    manifest = {
        "schema_version": 1,
        "scope": "hsic_amf_quantized",
        "complete": True,
        "reference_revision": REVISION,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "floating_artifact_sha256": identity(base),
        "config": asdict(config),
        "calibration_sha256": digest_file(manifest_path),
        "allocation_sha256": digest_file(allocation_path),
        "runtime_binary_sha256": base["runtime_binary_sha256"],
        "quantizer_binary_sha256": digest_file(quantizer),
        "quantized_tensors": 224,
        "achieved_bpw": allocation["achieved_bpw"],
        "verified_gguf_tensor_types": tensor_types,
        "files": {name: digest_file(output / name) for name in WEIGHTS},
        "precision": "HSIC-mixed GGUF LLM projections; BF16 embeddings/output/other matrices; "
        "F32 patch/vectors; FP16 KV; author GGML arithmetic",
        "scope_exclusions": [
            "vision/projector/action-head quantization",
            "default quantizer embedding/output heuristics",
        ],
        "source": source_metadata(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "calibration-path", "floating-path", "runtime-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    experiment = ExperimentConfig.load(args.config)
    if experiment.backend != "openvla_oft" or experiment.method != "actquant":
        raise ValueError("Expected an ActQuant OFT experiment configuration")
    result = quantize(
        ActQuantConfig.parse(experiment.parameters),
        args.calibration_path,
        args.floating_path,
        args.runtime_path,
        args.output,
    )
    print(json.dumps({"complete": result["complete"], "achieved_bpw": result["achieved_bpw"]}))


if __name__ == "__main__":
    main()
