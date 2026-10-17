"""Checksummed, explicitly supplied CUTLASS runtime; no Torch import or auto-build."""

import ctypes
import hashlib
import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file

CUTLASS_COMMIT = "f7b19de32c5d1f3cedfc735c2849f12b537522ee"
CUTLASS_HEADERS_SHA256 = "d12bb4cef3d33cb5e034b403ef62287fe30219b37a1f5fec0bb1c328ae6577af"
ABI_VERSION = 1


def kernel_source():
    return Path(__file__).with_name("csrc") / "mixed_w4.cu"


def header_digest(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(digest_file(path)))
    return digest.hexdigest()


def read_runtime(root):
    if root is None:
        raise ValueError("Mixed W4 execution requires an explicit runtime path")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {
        "schema_version": 1,
        "abi_version": ABI_VERSION,
        "engine": "mixed_w4_cutlass_sm80",
        "cutlass_commit": CUTLASS_COMMIT,
        "cutlass_headers_sha256": CUTLASS_HEADERS_SHA256,
        "kernel_source_sha256": digest_file(kernel_source()),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Mixed W4 runtime source/version mismatch")
    names = ("mixed_w4.so", "CUTLASS_LICENSE.txt")
    if manifest.get("files") != {name: digest_file(root / name) for name in names}:
        raise ValueError("Mixed W4 runtime checksum mismatch")
    return manifest


class MixedW4Runtime:
    """One loaded library shared by a policy's operators, never a global patch."""

    def __init__(self, root):
        self.identity = read_runtime(root)
        self.library = ctypes.CDLL(str((Path(root) / "mixed_w4.so").resolve(strict=True)))
        version = self.library.mixed_w4_abi
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != ABI_VERSION:
            raise ValueError("Mixed W4 binary ABI mismatch")
        self.gemm = self.library.int4_gemm
        self.gemm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        self.gemm.restype = ctypes.c_int

    def metadata(self):
        return dict(self.identity)
