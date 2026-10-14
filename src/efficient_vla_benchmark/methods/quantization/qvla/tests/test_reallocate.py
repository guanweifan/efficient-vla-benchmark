import json

import pytest
import torch

from efficient_vla_benchmark.methods.quantization.qvla import reallocate as api
from efficient_vla_benchmark.methods.quantization.qvla.config import QVLAConfig


def test_reallocation_restricts_only_frozen_candidates(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    manifest = dict(config=QVLAConfig().calibration_parameters(), target_modules=["a"], source={})
    (root / "manifest.json").write_text(json.dumps(manifest))
    torch.save(
        {"a": {0: [5.0, 5.0], 4: [0.1, 0.2], 8: [0.01, 0.02], 16: [0.0, 0.0]}},
        root / "sensitivity.pt",
    )
    monkeypatch.setattr(api, "read_manifest", lambda root, config: manifest)
    monkeypatch.setattr(api, "source_metadata", lambda: {})
    out = tmp_path / "restricted"
    result = api.reallocate(root, out, 8, (4, 8, 16))
    assert result["config"]["candidate_bits"] == [4, 8, 16]
    assert min(json.loads((out / "gates.json").read_text())["a"]) >= 4
    assert result["additional_calibration_observations"] == 0
    with pytest.raises(ValueError, match="frozen sensitivity"):
        api.reallocate(root, tmp_path / "invalid", 4, (2, 4, 8, 16))
    assert not (tmp_path / "invalid").exists()
