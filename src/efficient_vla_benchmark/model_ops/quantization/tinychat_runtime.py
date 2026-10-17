"""Explicit stream-correct TinyChat derivative; never patches an installed engine."""

import importlib.util
import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file

AUTHOR_COMMIT = "d6e797a42b9ef7778de8ee2352116e0f48a78d61"
ENGINE = "tinychat_current_stream_sm86_v1"
BINARY = "evb_tinychat_stream.so"
LICENSE_SHA256 = "2af485d269071b583c62c20250a63349f8313312cc9df0bd9007ca39c6d57bd9"
SOURCE_HASHES = {
    "gemm/gemm_cuda.cu": "f653271ee5232623e90019f72b6cacb9c5169b6d63a5967348bd9f8199c11651",
    "gemm/gemm_cuda.h": "bb3715cb374a8923ec2325b491fdfd393b1818f71d2488d598935f413f04bac4",
    "gemm/semaphore.h": "c5c9faf58373cd88098b3aa3914da86bd48f96f5a15d552014807c75442e19ae",
    "gemv/gemv_cuda.cu": "a5aea22eb22fadd3d444f475bf8a05a9e7477c22dfa2f429865d90d5d5c817d6",
    "gemv/gemv_cuda.h": "021cd59bb54bae11503d7c712883345672faae7a4cae4c08cb8e1ad3d3e689ec",
    "dequantize.cuh": "b4337fb4781b83c05795fe5523009c66dc805f7a8af709cb764a7c13f056ea5e",
    "dispatch_utils.cuh": "91221648663e47630df9f949d5bf00551b1a7526bbeadfe28153bf9f771872ac",
}
DERIVED_HASHES = dict(
    SOURCE_HASHES,
    **{
        "gemm/gemm_cuda.cu": "f4121c237ca5494cc39149ee489c2f0a09e5b7fbc6486e2bd58b33d3a99a4e1a",
        "gemv/gemv_cuda.cu": "0d11964d79943070e884228b41bf9ec2ac7cdfcad7768d27fa08fd7f4d9fb112",
    },
)
BINDING = (
    '#include <torch/extension.h>\n#include "gemm/gemm_cuda.h"\n'
    '#include "gemv/gemv_cuda.h"\n'
    'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemm_forward_cuda_new", '
    '&gemm_forward_cuda_new);m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new);}\n'
)


def derive_sources(source, output):
    """Mechanically redirect launches, preserving author arithmetic and license."""
    source, output = Path(source), Path(output)
    kernels = source / "awq/kernels/csrc/quantization_new"
    if {name: digest_file(kernels / name) for name in SOURCE_HASHES} != SOURCE_HASHES:
        raise ValueError("Expected pinned TinyChat source files")
    if digest_file(source / "LICENSE") != LICENSE_SHA256:
        raise ValueError("TinyChat's MIT notice is required")
    for name in SOURCE_HASHES:
        text = (kernels / name).read_text()
        if name.endswith(".cu"):
            text = "#include <ATen/cuda/CUDAContext.h>\n" + text
            if name.startswith("gemm/"):
                old = "<<<num_blocks, threads_per_block, kSmemByteSize>>>"
                new = (
                    "<<<num_blocks, threads_per_block, kSmemByteSize, "
                    "at::cuda::getCurrentCUDAStream()>>>"
                )
                expected = 2
            else:
                old = "<<<num_blocks, num_threads>>>"
                new = "<<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>"
                expected = 8  # Seven launches and one commented historical launch.
            if text.count(old) != expected:
                raise ValueError("Unexpected TinyChat launch sites")
            text = text.replace(old, new)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    if {name: digest_file(output / name) for name in DERIVED_HASHES} != DERIVED_HASHES:
        raise ValueError("Stream derivative differs from the reviewed source")
    (output / "binding.cpp").write_text(BINDING)
    (output / "AUTHOR_LICENSE.txt").write_bytes((source / "LICENSE").read_bytes())


def read_runtime(root):
    if root is None:
        raise ValueError("DiT graphs require an explicit TinyChat stream --runtime-path")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = dict(
        schema_version=1,
        engine=ENGINE,
        author_commit=AUTHOR_COMMIT,
        source_files=SOURCE_HASHES,
        derived_files=DERIVED_HASHES,
        loader_source_sha256=digest_file(Path(__file__)),
        compute_capability=[8, 6],
    )
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("TinyChat stream runtime source/version mismatch")
    names = [*DERIVED_HASHES, "binding.cpp", "AUTHOR_LICENSE.txt", BINARY]
    if manifest.get("files") != {name: digest_file(root / name) for name in names}:
        raise ValueError("TinyChat stream runtime checksum mismatch")
    if (root / "binding.cpp").read_text() != BINDING:
        raise ValueError("TinyChat extension binding mismatch")
    if digest_file(root / "AUTHOR_LICENSE.txt") != LICENSE_SHA256:
        raise ValueError("TinyChat author license mismatch")
    return manifest


class TinyChatStreamRuntime:
    """One immutable compiled engine, shared only by its owning policy's operators."""

    def __init__(self, root):
        import torch

        self.identity = read_runtime(root)
        if (
            self.identity["torch_version"] != torch.__version__
            or self.identity["torch_cuda"] != torch.version.cuda
            or list(torch.cuda.get_device_capability()) != self.identity["compute_capability"]
        ):
            raise ValueError("TinyChat stream runtime environment/device mismatch")
        spec = importlib.util.spec_from_file_location(
            "evb_tinychat_stream", (Path(root) / BINARY).resolve(strict=True)
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def forward(self, packed, inputs):
        import torch

        rows = inputs.numel() // inputs.shape[-1]
        with torch.cuda.device(inputs.device):
            if rows < 8:
                output = self.module.gemv_forward_cuda_new(
                    inputs,
                    packed.qweight,
                    packed.scales,
                    packed.scaled_zeros,
                    rows,
                    packed.out_features,
                    packed.in_features,
                    packed.group_size,
                )
            else:
                output = self.module.gemm_forward_cuda_new(
                    inputs, packed.qweight, packed.scales, packed.scaled_zeros
                )
        return output if packed.bias is None else output + packed.bias

    def metadata(self):
        return dict(self.identity)
