import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative,source",
    [
        ("src/efficient_vla_benchmark/evaluation/bad.py", "import torch"),
        ("src/efficient_vla_benchmark/evaluation/bad.py", "from ..methods.caching import paper"),
        (
            "backends/example/src/example/bad.py",
            "from efficient_vla_benchmark.methods.caching.paper import method",
        ),
        ("src/efficient_vla_benchmark/methods/caching/a/common.py", "from ..b import method"),
        (
            "src/efficient_vla_benchmark/methods/caching/a/config.py",
            "import efficient_vla_backend_openvla_oft",
        ),
    ],
)
def test_static_boundary_checker_detects_forbidden_imports(tmp_path, relative, source):
    script = Path(__file__).resolve().parents[1] / "scripts/check_boundaries.py"
    spec = importlib.util.spec_from_file_location("boundary_checker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert module.violations(tmp_path)
