import numpy as np
import pytest
import torch

from ..config import EfficientVLAConfig
from ..implementations.openvla_oft.method import select_tokens


@pytest.mark.parametrize("fraction", [0, 0.5, 1])
def test_selection_matches_independent_numpy_equations(fraction):
    rng = np.random.default_rng(52)
    features, scores = rng.normal(size=(32, 12)), rng.random(32)
    order = np.argsort(-scores, kind="stable")
    expected = set(order[: 4 + int(fraction * 12)])
    norm = features / np.linalg.norm(features, axis=-1, keepdims=True)
    distances = 1 - (norm @ norm[order[:4]].T).max(-1)
    distances[list(expected)] = -np.inf
    expected.update(np.argsort(-distances, kind="stable")[: 16 - len(expected)])
    actual = select_tokens(torch.tensor(features), torch.tensor(scores), 16, 4, fraction)
    assert set(actual.nonzero().flatten().tolist()) == expected


def test_zero_features_and_full_retention():
    result = select_tokens(torch.zeros(16, 4), torch.zeros(16), 16, 4, 0.5)
    assert result.all()


def test_relevance_uses_visual_queries_and_text_columns_not_transpose():
    from types import SimpleNamespace

    from efficient_vla_backend_openvla_oft.decoder import DecoderState, TokenLayout

    from ..implementations.openvla_oft.method import EfficientVLA

    method = EfficientVLA.__new__(EfficientVLA)
    method.config = SimpleNamespace(
        pruning_layer=1, tokens_per_view=2, key_tokens_per_view=1, task_fraction=1
    )
    method.layer_sizes = []
    layout = TokenLayout(12, patches_per_view=3, action_tokens=1)
    state = DecoderState(layout, torch.arange(12), torch.randn(1, 12, 4))
    requested = method.probes(1, state)
    assert requested["visual"].tolist() == [1, 2, 3, 4, 5, 6]
    # Unrelated BOS/visual/action columns must not affect the task ranking.
    attention = torch.full((6, 12), 100.0)
    attention[:, layout.text_start : layout.text_end] = torch.tensor(
        [[0.05], [0.45], [0.15], [0.30], [0.10], [0.40]]
    )
    keep = method.after_layer(1, state, {"visual": attention})
    assert state.positions[keep & layout.visual_mask(state.positions)].tolist() == [2, 3, 4, 6]
    assert keep[~layout.visual_mask(state.positions)].all()


@pytest.mark.parametrize(
    "parameters",
    [
        {"tokens_per_view": 0},
        {"pruning_layer": 31},
        {"task_fraction": float("nan")},
        {"enabled": 1},
        {"tokens_per_view": 4},
        {"other": 2},
    ],
)
def test_bad_config(parameters):
    with pytest.raises(ValueError):
        EfficientVLAConfig.parse(parameters)


@pytest.mark.gpu
def test_real_oft_parity_pruning_reset(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import EfficientVLA

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for parameters in ({"enabled": False}, {"tokens_per_view": 256}):
        backend.attach_method(EfficientVLA(backend, EfficientVLAConfig.parse(parameters)))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    backend.attach_method(EfficientVLA(backend, EfficientVLAConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(samples[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 256
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), actual)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
