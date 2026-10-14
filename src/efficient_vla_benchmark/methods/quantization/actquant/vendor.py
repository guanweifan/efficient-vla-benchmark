"""Read pinned author tools from an explicitly supplied isolated runtime installation."""

import importlib.util
import sys
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file

REVISION = "b64791125070652fe6b554e244fe809c79ef5246"
SOURCES = {
    "tools/hsic/compute_hsic_tensor.py": (
        "087aa0824bb96a5dc34e511d3370fd76cdd386b9d2799c8ced21a24527854e62"
    ),
    "tools/hsic/hsic_utils.py": "3fc1b742697db2a6d327ded58e16fe72a1af1515a7b20c122462a174557e9fcb",
    "tools/hsic/allocate.py": "9a617f86e031c9de9ea5274e6a11f94840bd581dc7779cc539006f14148362c9",
    "tools/fisher-diag/compute_amf_lq.py": (
        "208c827f74e2186cb30f60f71ccb75da5622cf193b322e86bcc152c09513e3f7"
    ),
    "tools/fisher-diag/compute_fisher.py": (
        "73bb4b0bd7e45d727560c73272253c36d72caf69bf7bf7dea7c40dafc8237eca"
    ),
}


def reference_source(runtime_path):
    if runtime_path is None:
        raise ValueError("ActQuant requires --runtime-path")
    source = Path(runtime_path) / "source"
    for name, expected in SOURCES.items():
        if not (source / name).is_file() or digest_file(source / name) != expected:
            raise ValueError("ActQuant author source differs from the pinned revision")
    return source


def author_module(runtime_path, relative):
    source = reference_source(runtime_path)
    if relative not in SOURCES:
        raise ValueError("Unregistered author tool")
    name = "_evb_actquant_" + Path(relative).stem
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != (source / relative).resolve():
            raise ValueError("Conflicting ActQuant source imports")
        return module
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(source / "gguf-py"))
        sys.path.insert(0, str(source / "tools/fisher-diag"))
        sys.path.insert(0, str(source / "tools/hsic"))
        spec = importlib.util.spec_from_file_location(name, source / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def gguf_module(runtime_path):
    """Use the exact reader shipped with the pinned quantizer, not another install."""
    source = reference_source(runtime_path)
    author_module(runtime_path, "tools/fisher-diag/compute_fisher.py")
    module = sys.modules["gguf"]
    if not Path(module.__file__).resolve().is_relative_to((source / "gguf-py").resolve()):
        raise ValueError("The GGUF Python reader is not from the pinned runtime")
    return module
