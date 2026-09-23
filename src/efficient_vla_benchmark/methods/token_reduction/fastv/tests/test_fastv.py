import numpy as np
import pytest
import torch

from ..config import FastVConfig
from ..selection import attention_selection


def test_selection_protects_nonvisual_and_breaks_ties_in_position_order():
    scores = torch.tensor([99.0, 0.2, 0.9, 0.9, 0.1, 100.0])
    visual = torch.tensor([False, True, True, True, True, False])
    assert attention_selection(scores, visual, 0.5).tolist() == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert attention_selection(scores, visual, 1).all()


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"keep_ratio": 0},
        {"keep_ratio": float("nan")},
        {"keep_ratio": True},
        {"pruning_layer": 0},
        {"unknown": 1},
    ],
)
def test_config_rejects_invalid(parameters):
    with pytest.raises(ValueError):
        FastVConfig.parse(parameters)


@pytest.mark.gpu
def test_real_oft_selection_parity_reset_detach(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import FastV

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    for config in (FastVConfig(enabled=False), FastVConfig(keep_ratio=1)):
        backend.attach_method(FastV(backend, config))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    backend.attach_method(FastV(backend, FastVConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(observations[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 256
    lengths = backend.query_diagnostics["layer_sequence_lengths"]
    assert lengths[:2] == [lengths[0]] * 2
    assert lengths[2:] == [lengths[0] - 256] * 30
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
