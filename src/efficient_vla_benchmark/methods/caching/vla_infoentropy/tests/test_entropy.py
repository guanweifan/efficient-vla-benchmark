import numpy as np
import pytest
import torch

from ..config import InfoEntropyConfig
from ..entropy import image_entropy, quotas
from ..implementations.openvla_oft.method import attention_information


def test_histogram_entropy_against_explicit_patch_counts():
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    actual = image_entropy(image)
    gray = np.rint(image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    for row, col in ((0, 0), (5, 7), (15, 15)):
        values, counts = np.unique(
            gray[row * 14 : (row + 1) * 14, col * 14 : (col + 1) * 14], return_counts=True
        )
        p = counts / 196
        assert actual[row * 16 + col] == pytest.approx(-(p * np.log2(p)).sum() / 8)
    assert not image_entropy(np.zeros_like(image)).any()


def test_time_quota_uses_fixed_horizon_and_clamps():
    assert quotas(0, 40, 60, 100) == (60, 40)
    assert quotas(50, 40, 60, 100) == (50, 50)
    assert quotas(1000, 40, 60, 100) == (40, 60)


def test_attention_entropy_direction_and_small_information():
    attention = torch.tensor([[0.01, 0.0], [0.01, 0.02]])
    result = attention_information(attention)
    assert result[0] == 0 and 0 < result[1] < 1e-3
    assert not attention_information(torch.ones(1, 2)).any()


def test_attention_information_matches_numpy_equations_four_to_six():
    rng = np.random.default_rng(53)
    attention = rng.uniform(0, 0.1, (2, 3, 5, 16))
    mean = attention.mean((0, 1))
    exp = np.exp(mean - mean.max(0, keepdims=True))
    probability = exp / exp.sum(0, keepdims=True)
    expected = 1 + (probability * np.log2(probability)).sum(0) / np.log2(5)
    actual = attention_information(torch.tensor(mean))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-8, atol=1e-14)
    # A transpose or omission of the second softmax changes this distribution.
    raw_probability = mean / mean.sum(0, keepdims=True)
    wrong = 1 + (raw_probability * np.log2(raw_probability)).sum(0) / np.log2(5)
    assert not np.allclose(actual.numpy(), wrong)


def test_all_quota_steps_match_literal_floor_schedule():
    import math

    for step in range(121):
        alpha = min(step / 100, 1)
        assert quotas(step, 40, 60, 100) == (
            math.floor(60 - 20 * alpha),
            math.floor(40 + 20 * alpha),
        )
    assert quotas(-1, 40, 60, 100) == (60, 40)


@pytest.mark.parametrize(
    "params",
    [
        {"enabled": 1},
        {"transition_steps": 0},
        {"low_quota": 100},
        {"reuse_layers": [3, 2]},
        {"high_quota": 300},
    ],
)
def test_invalid_config(params):
    with pytest.raises(ValueError):
        InfoEntropyConfig.parse(params)


@pytest.mark.gpu
def test_real_oft_disabled_and_sdpa_cold_parity_warm_cache(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import VLAInfoEntropy

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    backend.attach_method(VLAInfoEntropy(backend, InfoEntropyConfig.parse({"enabled": False})))
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    backend.detach_method()
    backend.attach_method(
        VLAInfoEntropy(backend, InfoEntropyConfig.parse({"attention_backend": "sdpa"}))
    )
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), expected, rtol=0, atol=1e-6)
    assert backend.query_diagnostics["cache_cold"]
    backend.method.before_plan(samples[0], 50)
    actual = backend.predict_chunk(samples[0])
    assert np.isfinite(actual).all() and actual.shape == (8, 7)
    assert not backend.query_diagnostics["cache_cold"]
    assert backend.query_diagnostics["reused_tokens_last_layer"] > 0
    assert backend.query_diagnostics["visual_quota"] == 50
    assert all(
        value.shape[2] == backend.method.last_layout.length
        for value in backend.method.store.keys.values()
    )
    backend.reset(context)
    assert not backend.method.store.keys
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), expected, rtol=0, atol=1e-6)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
