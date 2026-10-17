"""Build the optional integer runtime from caller-supplied pinned CUTLASS headers."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.model_ops.quantization.mixed_runtime import (
    ABI_VERSION,
    CUTLASS_COMMIT,
    CUTLASS_HEADERS_SHA256,
    MixedW4Runtime,
    header_digest,
    kernel_source,
)


def build(cutlass, output, nvcc):
    if output.exists():
        raise ValueError("Choose a new runtime output; never overwrite a measured binary")
    if header_digest(cutlass / "include") != CUTLASS_HEADERS_SHA256:
        raise ValueError("Expected unmodified CUTLASS 3.5.1 pinned headers")
    license_file = cutlass / "LICENSE.txt"
    if not license_file.is_file():
        raise ValueError("CUTLASS license must accompany its source")
    compiler = subprocess.check_output([str(nvcc), "--version"], text=True).strip()
    output.mkdir(parents=True)
    flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-shared",
        "-Xcompiler=-fPIC",
        "-gencode=arch=compute_80,code=sm_80",
    ]
    subprocess.run(
        [
            str(nvcc),
            *flags,
            "-I",
            str(cutlass / "include"),
            str(kernel_source()),
            "-o",
            str(output / "mixed_w4.so"),
        ],
        check=True,
    )
    shutil.copyfile(license_file, output / "CUTLASS_LICENSE.txt")
    manifest = dict(
        schema_version=1,
        abi_version=ABI_VERSION,
        engine="mixed_w4_cutlass_sm80",
        cutlass_commit=CUTLASS_COMMIT,
        cutlass_headers_sha256=CUTLASS_HEADERS_SHA256,
        kernel_source_sha256=digest_file(kernel_source()),
        compiler=compiler,
        compiler_flags=flags,
        files={name: digest_file(output / name) for name in ("mixed_w4.so", "CUTLASS_LICENSE.txt")},
    )
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    MixedW4Runtime(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.cutlass_path, args.output_dir, args.nvcc), indent=2))
