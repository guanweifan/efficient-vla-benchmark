import numpy as np
import pytest
import torch

from ..config import DivPruneConfig
from ..selection import max_min_selection, visual_selection


def test_matches_independent_numpy_greedy_reference():
    rng = np.random.default_rng(31)
    features = rng.normal(size=(19, 11)).astype(np.float32)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    distance = np.maximum(0, 1 - unit @ unit.T)
    first = distance.copy()
    np.fill_diagonal(first, np.inf)
    chosen = [int(first.min(0).argmax())]
    for _ in range(7):
        score = distance[chosen].min(0)
        score[chosen] = -np.inf
        chosen.append(int(score.argmax()))
    assert max_min_selection(torch.tensor(features), 8).tolist() == sorted(chosen)


def test_identical_zero_and_singleton_features_never_duplicate():
    for features in (torch.ones(8, 3), torch.zeros(8, 3)):
        assert max_min_selection(features, 5).tolist() == list(range(5))
    assert max_min_selection(torch.ones(1, 3), 1).tolist() == [0]


def test_nonvisual_positions_are_protected():
    visual = torch.tensor([False, True, True, True, True, False])
    keep = visual_selection(torch.randn(6, 8), visual, 0.5)
    assert keep[~visual].all() and keep[visual].sum() == 2


@pytest.mark.parametrize(
    "params", [{"keep_ratio": 0}, {"enabled": 1}, {"keep_ratio": float("nan")}, {"foo": 1}]
)
def test_invalid_config(params):
    with pytest.raises(ValueError):
        DivPruneConfig.parse(params)


@pytest.mark.gpu
def test_real_oft_parity_selection_reset_detach(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import DivPrune

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    for config in (DivPruneConfig(enabled=False), DivPruneConfig(keep_ratio=1)):
        backend.attach_method(DivPrune(backend, config))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    backend.attach_method(DivPrune(backend, DivPruneConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(observations[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 256
    lengths = backend.query_diagnostics["layer_sequence_lengths"]
    assert lengths == [lengths[0]] * 32
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
