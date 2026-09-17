import ast
from pathlib import Path

import numpy as np
import pytest

from efficient_vla_benchmark.methods.token_reduction.safe_pruner.config import SAFEPrunerConfig


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"keep_ratio": 0},
        {"pruning_layer": 1},
        {"reference_layers": [0]},
        {"reference_layers": [0, 0, 31]},
        {"reference_layers": [1, 31]},
        {"similarity_threshold": 2},
        {"forecast_weight": float("nan")},
        {"forecast_weight": -1},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        SAFEPrunerConfig.parse(bad)


def test_correspondence_and_fusion_match_reference():
    import torch

    from ..implementations.openvla_oft.method import (
        correspondences,
        fused_selection,
        transfer_saliency,
    )

    root = Path(__file__).resolve().parents[6]
    source = (
        root
        / "third_party/methods/safe_pruner/src/openvla-oft"
        / "prismatic/extern/hf/modeling_prismatic.py"
    )
    if not source.exists():
        pytest.skip("Initialize pinned SAFE-Pruner source")
    tree = ast.parse(source.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "find_corresponding_tokens_optimized"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"
        ),
        namespace,
    )
    torch.manual_seed(15)
    reference = torch.randn(32, 16)
    current = reference.flip(0) + torch.randn(32, 16) * 0.01
    expected = namespace["find_corresponding_tokens_optimized"](current, reference)[:, 1].long()
    torch.testing.assert_close(correspondences(current, reference), expected, rtol=0, atol=0)
    saliency = torch.rand(2, 32)
    forecast = transfer_saliency(current, reference, saliency)
    torch.testing.assert_close(forecast, saliency[:, expected])
    shallow = torch.rand(32)
    torch.testing.assert_close(
        fused_selection(shallow, forecast, 0.5, 8),
        (shallow + 0.5 * forecast.mean(0)).topk(8).indices,
    )
    with pytest.raises(ValueError, match="patch IDs"):
        transfer_saliency(current, reference, saliency[:, :-1])


@pytest.mark.gpu
def test_real_oft_refresh_prune_disabled_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import SAFEPruner

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for params in ({"enabled": False}, {"keep_ratio": 1}):
        backend.attach_method(SAFEPruner(backend, SAFEPrunerConfig.parse(params)))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    method = SAFEPruner(backend, SAFEPrunerConfig())
    backend.attach_method(method)
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    assert backend.query_diagnostics["keyframe"]
    assert method.reference_features.shape[0] == 512
    assert method.reference_saliency.shape == (2, 512)
    actual = backend.predict_chunk(samples[0])
    assert np.isfinite(actual).all() and actual.shape == (8, 7)
    assert not backend.query_diagnostics["keyframe"]
    assert backend.query_diagnostics["visual_tokens_final"] == 128
    method.reference_saliency = method.reference_saliency.clone().zero_()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    assert backend.query_diagnostics["keyframe"]
    assert backend.query_diagnostics["refreshes"] == 2
    backend.reset(context)
    assert method.reference_features is None
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
