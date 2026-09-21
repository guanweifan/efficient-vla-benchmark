import numpy as np
import pytest

from efficient_vla_benchmark.methods.token_reduction.vla_iap.config import VLAIAPConfig


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"temperature": 0},
        {"history_decay": 1},
        {"iou_threshold": -1},
        {"background_std_scale": 0},
        {"closing_kernel": 2},
        {"gaussian_sigma": 0},
        {"edge_weight": float("nan")},
        {"epsilon": 0},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        VLAIAPConfig.parse(bad)


def test_sobel_and_constant_maps():
    import torch

    from ..implementations.openvla_oft.method import minmax, sobel_prior

    image = torch.ones(2, 3, 224, 224)
    assert not sobel_prior(image, 1e-6).any()
    image[:, :, :, 112:] = 0
    edge = sobel_prior(image, 1e-6)
    assert edge[:, :, 7:9].min() > 0.9
    assert not edge[:, :, :7].any() and not edge[:, :, 9:].any()
    assert not minmax(torch.ones(2, 16, 16), 1e-6).any()


def test_modes_and_geometric_rescue():
    import torch

    from ..implementations.openvla_oft.method import interaction_selection

    cfg = VLAIAPConfig(geometry_threshold=1.5, semantic_radius=1)
    semantic = torch.zeros(2, 16, 16)
    motion, edge = semantic.clone(), semantic.clone()
    semantic[:, 2:4, 2:4] = 1
    motion[0, 2:4, 2:4] = 1
    motion[1, 10:12, 10:12] = 1
    edge[:, 15, 15] = 2
    selected, iou, aggressive = interaction_selection(semantic, motion, edge, cfg)
    assert aggressive.tolist() == [True, False]
    assert iou.tolist() == [1.0, 0.0]
    assert selected.reshape(2, 16, 16)[:, 15, 15].all()
    assert selected.reshape(2, 16, 16)[0, 2:4, 2:4].all()
    assert not selected.reshape(2, 16, 16)[0, 8, 8]
    # With a very sparse signal, mean - .5*std is negative: zero-valued
    # background is NOT below that threshold and conservative mode keeps it.
    assert selected.reshape(2, 16, 16)[1, 8, 8]
    zeros = torch.zeros_like(semantic)
    selected, iou, aggressive = interaction_selection(zeros, zeros, zeros, cfg)
    assert selected.all() and not aggressive.any() and not iou.any()


def test_temporal_equation_and_semantic_finiteness():
    import torch

    from ..implementations.openvla_oft.method import motion_prior, semantic_prior

    cfg = VLAIAPConfig()
    torch.manual_seed(13)
    current, previous, old = (torch.randn(2, 256, 8) for _ in range(3))
    history = torch.rand(2, 16, 16)
    motion, updated = motion_prior(current, previous, old, history, cfg)
    expected = 0.3 * (current - 2 * previous + old).norm(dim=-1).reshape(2, 16, 16) + 0.7 * history
    torch.testing.assert_close(updated, expected)
    assert torch.isfinite(motion).all() and motion.min() >= 0 and motion.max() <= 1
    semantic = semantic_prior(current, torch.randn(4, 8), cfg)
    assert torch.isfinite(semantic).all() and semantic.min() >= 0 and semantic.max() <= 1


def test_semantic_prior_matches_numpy_under_disclosed_alignment_convention():
    import torch

    from ..implementations.openvla_oft.method import semantic_prior

    rng = np.random.default_rng(29)
    visual, text = rng.normal(size=(2, 256, 8)), rng.normal(size=(5, 8))
    cfg = VLAIAPConfig()
    center = visual.mean(1, keepdims=True)
    v, t = visual - center, text.mean(0)[None, None] - center
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), cfg.epsilon)
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), cfg.epsilon)
    logits = (v * t).sum(-1) / cfg.temperature
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    p = (exp / exp.sum(-1, keepdims=True)).reshape(2, 16, 16)
    pooled = np.zeros_like(p)
    for row in range(16):
        for col in range(16):
            pooled[:, row, col] = p[:, max(0, row - 1) : row + 2, max(0, col - 1) : col + 2].mean(
                (1, 2)
            )
    low, high = pooled.min((1, 2), keepdims=True), pooled.max((1, 2), keepdims=True)
    expected = (pooled - low) / np.maximum(high - low, cfg.epsilon)
    actual = semantic_prior(torch.tensor(visual), torch.tensor(text), cfg)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-6)
    # Shared centering must be invariant to a common feature-space translation.
    shift = torch.arange(8).float()
    shifted = semantic_prior(
        torch.tensor(visual).float() + shift, torch.tensor(text).float() + shift, cfg
    )
    torch.testing.assert_close(actual, shifted, rtol=2e-4, atol=1e-5)


def test_motion_filters_linear_feature_drift_without_future_frames():
    import torch

    from ..implementations.openvla_oft.method import motion_prior

    old = torch.arange(2 * 256 * 4).reshape(2, 256, 4).float()
    previous, current = old + 2, old + 4
    motion, history = motion_prior(current, previous, old, torch.zeros(2, 16, 16), VLAIAPConfig())
    assert not motion.any() and not history.any()


def test_iou_equality_uses_conservative_branch_and_no_forced_ratio():
    import torch

    from ..implementations.openvla_oft.method import interaction_selection

    semantic = torch.zeros(2, 16, 16)
    semantic[:, 2:4, 2:4] = 1
    # Exact equality at theta_iou=1 belongs to Eq. 12's conservative branch.
    selected, iou, aggressive = interaction_selection(
        semantic, semantic, torch.zeros_like(semantic), VLAIAPConfig(iou_threshold=1)
    )
    assert iou.tolist() == [1.0, 1.0]
    assert not aggressive.any()
    assert selected.all()


@pytest.mark.gpu
def test_real_oft_warmup_history_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import VLAIAP

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    second = backend.predict_chunk(samples[8])
    backend.attach_method(VLAIAP(backend, VLAIAPConfig(enabled=False)))
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    backend.detach_method()
    method = VLAIAP(backend, VLAIAPConfig())
    backend.attach_method(method)
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    np.testing.assert_array_equal(backend.predict_chunk(samples[8]), second)
    actual = backend.predict_chunk(samples[16])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert not backend.query_diagnostics["warmup"]
    assert 2 <= backend.query_diagnostics["visual_tokens_final"] <= 512
    assert len(method.history) == 2 and method.history[0].shape == (2, 256, 4096)
    assert method.edge is None
    backend.reset(context)
    assert not method.history and method.motion_history is None
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
    backend.predict_chunk(samples[8])
    np.testing.assert_array_equal(backend.predict_chunk(samples[16]), actual)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
