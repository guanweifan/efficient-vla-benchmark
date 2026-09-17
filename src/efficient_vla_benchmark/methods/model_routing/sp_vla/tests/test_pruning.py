import numpy as np
import pytest
import torch

from ..config import SPVLAConfig
from ..implementations.openvla_oft.method import attention_mass, edge_tokens, semantic_selection


def test_released_mass_rule_is_not_inverted():
    assert attention_mass(0, 0.5, 0.9) == 1
    assert attention_mass(0.5, 0.5, 0.9) == 1
    assert attention_mass(1, 0.5, 0.9) == pytest.approx(0.1)
    assert attention_mass(2, 0.5, 0.9) == pytest.approx(0.1)


def test_edges_preserve_camera_patch_geometry():
    image = np.zeros((224, 224, 3), np.uint8)
    image[70:98, 112:140] = 255
    mask = edge_tokens(image, 100, 200).reshape(16, 16)
    assert mask.sum() > 0 and mask[4:8, 7:11].sum() == mask.sum()
    assert not edge_tokens(np.zeros_like(image), 100, 200).any()


def test_semantic_mass_retains_crossing_token():
    q = torch.ones(1, 1, 4, 1)
    k = torch.tensor([[[[4.0], [3.0], [2.0], [1.0]]]])
    result = semantic_selection(q, k, 0.7)
    assert result.tolist() == [True, True, False, False]
    assert semantic_selection(q, k, 1).all()


@pytest.mark.parametrize(
    "params",
    [
        {"scheduling": 1},
        {"token_pruning": 1},
        {"z_prune_threshold": 1},
        {"max_prune_fraction": -1},
        {"canny_low": 201},
    ],
)
def test_invalid_pruning_config(params):
    with pytest.raises(ValueError):
        SPVLAConfig.parse(params)


@pytest.mark.gpu
def test_real_oft_dense_parity_pruning_and_cleanup(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import SPVLA

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    backend.attach_method(SPVLA(backend, SPVLAConfig.parse({"token_pruning": True})))
    backend.reset(context)
    # Cold/low-Z still probes attention, but must preserve native predictions.
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    assert backend.query_diagnostics["visual_tokens_final"] == 512
    action = np.array([0, 0, 1, 0, 0, 0, 1.0])
    backend.method.after_action(action, 0)
    actual = backend.predict_chunk(samples[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] < 512
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
