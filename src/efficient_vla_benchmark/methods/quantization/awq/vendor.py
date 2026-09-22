"""Import pinned AWQ primitives without installing its conflicting application stack."""

import importlib
import importlib.machinery
import importlib.util
import sys

from efficient_vla_benchmark.core.resources import project_source


def reference_module(name: str):
    namespace = "_evb_pinned_awq"
    if name not in {"quantize.qmodule", "quantize.quantizer", "quantize.auto_clip"}:
        raise ValueError("Unsupported AWQ reference module")
    root = project_source("third_party/runtimes/llm_awq") / "awq"
    if not (root / "quantize" / "qmodule.py").is_file():
        raise RuntimeError("Initialize the pinned llm-awq submodule first")
    # Do not execute the reference quantize/__init__.py: it eagerly imports an
    # unrelated W8A8 application and its absolute `awq` package dependencies.
    for suffix in ("", ".quantize", ".utils"):
        package = namespace + suffix
        if package not in sys.modules:
            directory = root if not suffix else root / suffix[1:]
            spec = importlib.machinery.ModuleSpec(package, loader=None, is_package=True)
            spec.submodule_search_locations = [str(directory)]
            sys.modules[package] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{namespace}.{name}")
