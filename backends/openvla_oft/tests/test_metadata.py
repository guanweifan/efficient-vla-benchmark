from types import SimpleNamespace

import pytest
from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend


@pytest.mark.parametrize("selected", [None, "sdpa", "eager"])
def test_metadata_reports_selected_attention_without_overwriting_native_config(selected):
    backend = object.__new__(OpenVLAOFTBackend)
    decoder = SimpleNamespace()
    if selected is not None:
        decoder._evb_adapter = SimpleNamespace(strategy=SimpleNamespace(attention_backend=selected))
    backend.model = SimpleNamespace(
        language_model=SimpleNamespace(model=decoder),
        config=SimpleNamespace(_attn_implementation="sdpa"),
    )
    backend.method, backend.method_name, backend.precision = None, "none", "bfloat16"
    backend._model_manifest = {"sha256": "test"}
    data = backend.metadata()
    assert data["attention_implementation"] == (selected or "sdpa")
    assert data["native_attention_implementation"] == "sdpa"
