import json
from dataclasses import asdict

import pytest

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.methods.quantization.actquant import artifact
from efficient_vla_benchmark.methods.quantization.actquant.config import (
    ActQuantConfig,
    ActQuantFloatingConfig,
)
from efficient_vla_benchmark.methods.quantization.actquant.vendor import REVISION


def setup(tmp_path, monkeypatch, quantized=False):
    root, runtime = tmp_path / "artifact", tmp_path / "runtime"
    root.mkdir()
    (runtime / "build/bin").mkdir(parents=True)
    binary = runtime / "build/bin/openvla_oft.so"
    binary.write_bytes(b"test binary")
    for name in artifact.WEIGHTS:
        (root / name).write_bytes(name.encode())
    monkeypatch.setattr(artifact, "reference_source", lambda path: path / "source")
    config = ActQuantConfig() if quantized else ActQuantFloatingConfig()
    manifest = dict(
        schema_version=1,
        complete=True,
        reference_revision=REVISION,
        scope="hsic_amf_quantized" if quantized else "floating_runtime_control",
        runtime_binary_sha256=digest_file(binary),
        files={n: digest_file(root / n) for n in artifact.WEIGHTS},
    )
    if quantized:
        (root / "allocation.json").write_text("{}")
        manifest.update(
            config=asdict(config),
            calibration_sha256="a" * 64,
            quantized_tensors=224,
            allocation_sha256=digest_file(root / "allocation.json"),
        )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, runtime, config


@pytest.mark.parametrize("quantized", [False, True])
def test_complete_artifact_and_tamper_detection(tmp_path, monkeypatch, quantized):
    root, runtime, config = setup(tmp_path, monkeypatch, quantized)
    assert artifact.read_manifest(root, config, runtime)["complete"]
    (root / "llm.gguf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        artifact.read_manifest(root, config, runtime)


def test_floating_control_cannot_impersonate_calibrated_method(tmp_path, monkeypatch):
    root, runtime, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identity"):
        artifact.read_manifest(root, ActQuantConfig(), runtime)


def test_runtime_binary_is_part_of_identity(tmp_path, monkeypatch):
    root, runtime, config = setup(tmp_path, monkeypatch)
    (runtime / "build/bin/openvla_oft.so").write_bytes(b"other build")
    with pytest.raises(ValueError, match="runtime"):
        artifact.read_manifest(root, config, runtime)
