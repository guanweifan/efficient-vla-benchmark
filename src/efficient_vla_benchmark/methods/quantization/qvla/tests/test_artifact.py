from types import SimpleNamespace

import pytest
import torch

from efficient_vla_benchmark.methods.quantization.qvla.artifact import validate_gates
from efficient_vla_benchmark.methods.quantization.qvla.config import QVLAConfig


def fixture():
    manifest = {
        "target_modules": ["layer"],
        "inactive_target_modules": [],
        "stats": {"final_avg_bits": 4.0},
    }
    modules = {"layer": SimpleNamespace(weight=torch.empty(2, 4))}
    return manifest, modules


@pytest.mark.parametrize("row", [[4.0, 4.0], [True, 4], [[4], 4], [16, 16], [4]])
def test_invalid_gates_rejected(row):
    manifest, modules = fixture()
    with pytest.raises(ValueError):
        validate_gates(manifest, {"layer": row}, modules, QVLAConfig())


def test_valid_gates_and_inactive_overlap():
    manifest, modules = fixture()
    assert len(validate_gates(manifest, {"layer": [4, 4]}, modules, QVLAConfig())) == 64
    manifest["inactive_target_modules"] = ["layer"]
    with pytest.raises(ValueError):
        validate_gates(manifest, {"layer": [4, 4]}, modules, QVLAConfig())
