import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from efficient_vla_benchmark.methods.token_reduction.vla_pruner.config import VLAPrunerConfig


@pytest.mark.parametrize(
    "bad",
    [
        {"keep_ratio": 0},
        {"enabled": 1},
        {"decay": float("inf")},
        {"history_size": 0},
        {"pruning_layer": True},
        {"pruning_layer": -1},
        {"x": 2},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        VLAPrunerConfig.parse(bad)


def test_dual_score_selection_matches_pinned_source():
    import torch

    from ..implementations.openvla_oft.method import combine_filter

    root = Path(__file__).resolve().parents[6]
    source = (
        root
        / "third_party/methods/vla_pruner/src/openvla-oft/transformers"
        / "src/transformers/models/llama/modeling_llama.py"
    )
    if not source.exists():
        pytest.skip("Initialize pinned VLA-Pruner source")
    tree = ast.parse(source.read_text())
    names = ("_redundancy_minization", "_vlapruner_image_indices")
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "Optional": Optional}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"
        ),
        namespace,
    )
    dummy = SimpleNamespace(_slice_range=lambda a, b, n: (max(0, min(a, n)), max(0, min(b, n))))
    dummy._redundancy_minization = lambda x, k: namespace["_redundancy_minization"](dummy, x, k)
    torch.manual_seed(13)
    for count in (1, 4, 8, 16):
        attention = torch.softmax(torch.randn(30, 30), -1)
        embeddings = torch.randn(1, 30, 32)
        cfg = {
            "vla_pruner_mode": "semantic_action",
            "action_token_start": 23,
            "action_token_end": 29,
        }
        expected, _ = namespace["_vlapruner_image_indices"](
            dummy, attention, 1, 17, 30, count, cfg, embeddings
        )
        actual = (
            combine_filter(
                attention[:23, 1:17].mean(0),
                attention[23:29, 1:17].mean(0),
                embeddings[0, 1:17],
                count,
            )
            + 1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_degenerate_features_have_unique_budget_and_history_is_newest_first():
    import torch

    from ..implementations.openvla_oft.method import diverse_indices, temporal_scores

    ids = diverse_indices(torch.ones(8, 16), 5)
    assert ids.unique().numel() == 5
    history = deque([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])])
    torch.testing.assert_close(
        temporal_scores(history, 0.5, torch.float32), torch.tensor([1 / 3, 2 / 3])
    )


@pytest.mark.gpu
def test_real_oft_warm_start_active_pruning_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import VLAPruner

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for params in ({"enabled": False}, {"keep_ratio": 1}):
        backend.attach_method(VLAPruner(backend, VLAPrunerConfig.parse(params)))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    method = VLAPruner(backend, VLAPrunerConfig())
    backend.attach_method(method)
    backend.reset(context)
    for _ in range(3):
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        assert backend.query_diagnostics["warm_start"]
    for _ in range(2):
        actual = backend.predict_chunk(samples[0])
        assert actual.shape == (8, 7) and np.isfinite(actual).all()
        assert backend.query_diagnostics["visual_tokens_final"] == 128
        assert len(method.history) == 3
    assert (method.history[-1] == 0).sum() >= 384
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
