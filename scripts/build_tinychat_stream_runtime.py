"""Build the pinned stream-correct TinyChat derivative without changing its checkout."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.model_ops.quantization import tinychat_runtime as runtime


def build(source, output, nvcc, jobs):
    if output.exists():
        raise ValueError("Choose a new runtime artifact; never overwrite a measured binary")
    if not 1 <= jobs <= 16:
        raise ValueError("Build jobs must be in [1,16]")
    compiler = subprocess.check_output([str(nvcc), "--version"], text=True).strip()
    os.environ["CUDA_HOME"] = str(nvcc.resolve().parent.parent)
    os.environ["MAX_JOBS"] = str(jobs)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    import torch
    from torch.utils.cpp_extension import load

    output.mkdir(parents=True)
    runtime.derive_sources(source, output)
    flags = [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=2",
    ]
    load(
        name="evb_tinychat_stream",
        sources=[
            str(output / name) for name in ("binding.cpp", "gemm/gemm_cuda.cu", "gemv/gemv_cuda.cu")
        ],
        build_directory=str(output),
        extra_include_paths=[str(output.resolve())],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=flags,
        verbose=True,
    )
    names = [*runtime.DERIVED_HASHES, "binding.cpp", "AUTHOR_LICENSE.txt", runtime.BINARY]
    manifest = dict(
        schema_version=1,
        engine=runtime.ENGINE,
        author_commit=runtime.AUTHOR_COMMIT,
        source_files=runtime.SOURCE_HASHES,
        derived_files=runtime.DERIVED_HASHES,
        loader_source_sha256=digest_file(Path(runtime.__file__)),
        compute_capability=[8, 6],
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        compiler=compiler,
        compiler_flags=flags,
        files={name: digest_file(output / name) for name in names},
    )
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    runtime.read_runtime(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args.source_path, args.output_dir, args.nvcc, args.jobs), indent=2))
