import numpy as np
import pytest
import torch

from ..config import FlashVLAConfig
from ..scoring import information_scores


def test_ics_against_independent_numpy_svd():
    rng = np.random.default_rng(39)
    features = rng.normal(size=(16, 48)).astype(np.float32)
    left, values, _ = np.linalg.svd(features, full_matrices=False)
    expected = np.abs(left * values).sum(-1)
    np.testing.assert_allclose(
        information_scores(torch.tensor(features)).numpy(), expected, rtol=1e-5, atol=1e-5
    )


def test_zero_input_is_finite_zero():
    assert not information_scores(torch.zeros(16, 48)).any()


@pytest.mark.parametrize(
    "params",
    [
        {"enabled": 1},
        {"tokens_per_view": 0},
        {"pruning_layer": 31},
        {"action_reuse": True},
        {"svd_driver": "gesvda"},
        {"svd_driver": None},
        {"svd_driver": []},
    ],
)
def test_invalid_config(params):
    with pytest.raises(ValueError):
        FlashVLAConfig.parse(params)


@pytest.mark.parametrize("driver", ["auto", "gesvd", "gesvdj"])
def test_cpu_score_keeps_all_singular_directions(driver):
    # Orthogonal rows have known scores, including small but nonzero directions.
    matrix = torch.zeros(4, 8)
    matrix[:, :4] = torch.diag(torch.tensor([8.0, 3.0, 0.5, 0.125]))
    torch.testing.assert_close(
        information_scores(matrix, driver), torch.tensor([8.0, 3.0, 0.5, 0.125])
    )
    with pytest.raises(ValueError):
        information_scores(matrix, "gesvda")


@pytest.mark.gpu
def test_real_oft_disabled_pruning_reset_and_restore(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import FlashVLA

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for params in ({"enabled": False}, {"tokens_per_view": 256}):
        backend.attach_method(FlashVLA(backend, FlashVLAConfig.parse(params)))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    backend.attach_method(FlashVLA(backend, FlashVLAConfig()))
    backend.reset(context)
    actual = backend.predict_chunk(samples[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 320
    assert backend.method.attention_features is None
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), actual)
    backend.detach_method()
    backend.attach_method(FlashVLA(backend, FlashVLAConfig(svd_driver="gesvd")))
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), actual, rtol=0, atol=1e-6)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
