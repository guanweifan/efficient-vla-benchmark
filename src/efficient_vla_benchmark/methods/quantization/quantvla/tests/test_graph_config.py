import pytest

from efficient_vla_benchmark.methods.quantization.quantvla.config import QuantVLAConfig


def test_graph_execution_preserves_frozen_fp16_calibration_identity():
    eager = QuantVLAConfig.parse({"runtime": "tinychat_fp16"})
    graph = QuantVLAConfig.parse({"runtime": "tinychat_fp16", "dit_graphs": True})
    assert graph.calibration_parameters() == eager.calibration_parameters()
    assert "dit_graphs" not in graph.calibration_parameters()
    for invalid in ({"dit_graphs": True}, {"runtime": "tinychat_fp16", "dit_graphs": 1}):
        with pytest.raises(ValueError, match="graphs"):
            QuantVLAConfig.parse(invalid)
