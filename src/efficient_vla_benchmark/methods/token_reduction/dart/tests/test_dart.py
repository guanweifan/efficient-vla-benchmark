import numpy as np
import pytest
import torch

from ..config import DARTConfig
from ..selection import pivot_selection


def test_pivots_and_sequential_selection_against_numpy():
    rng = np.random.default_rng(42)
    h, k = (
        rng.normal(size=(24, 11)).astype(np.float32),
        rng.normal(size=(24, 11)).astype(np.float32),
    )
    visual, text = np.zeros(24, bool), np.zeros(24, bool)
    visual[1:17], text[18:23] = True, True
    norm = np.abs(k).sum(-1)
    vp = np.flatnonzero(visual)[np.argsort(-norm[visual], kind="stable")[:2]]
    tp = np.flatnonzero(text)[np.argsort(-norm[text], kind="stable")[:2]]
    selected = set(vp.tolist())
    unit = h / np.linalg.norm(h, axis=1, keepdims=True)
    for pivot in sorted([*vp, *tp]):
        available = sorted(set(np.flatnonzero(visual)) - selected)
        score = unit[available] @ unit[pivot]
        selected.update(np.array(available)[np.argsort(score, kind="stable")[:2]].tolist())
    expected = ~visual
    expected[list(selected)] = True
    actual = pivot_selection(
        torch.tensor(h), torch.tensor(k), torch.tensor(visual), torch.tensor(text), 0.5, 2, 2
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert int(actual[torch.tensor(visual)].sum()) == 10


@pytest.mark.parametrize(
    "params", [{"pruning_layer": 0}, {"text_pivots": 0}, {"keep_ratio": True}, {"unknown": 1}]
)
def test_invalid_config(params):
    with pytest.raises(ValueError):
        DARTConfig.parse(params)


@pytest.mark.gpu
def test_real_oft_parity_selection_reset_detach(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import DART

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    for config in (DARTConfig(enabled=False), DARTConfig(keep_ratio=1)):
        backend.attach_method(DART(backend, config))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    backend.attach_method(DART(backend, DARTConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(observations[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 260
    assert backend.method.keys is None
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
