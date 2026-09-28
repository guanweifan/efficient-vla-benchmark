"""Independent saliency/diversity equations and native CLS feature provenance."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import AMMRConfig
from ..selection import select_current_frame
from ..strategy import CurrentFrameStrategy


def numpy_reference(patches, cls, target, *, clip_negative=False):
    features = patches.astype(np.float64)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    cls = cls.astype(np.float64) / max(np.linalg.norm(cls), 1e-12)
    saliency = features @ cls
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    selected = []
    while len(selected) < target:
        values = []
        for index in range(len(features)):
            if index in selected:
                values.append(-np.inf)
                continue
            similarity = (
                max(float(features[index] @ features[j]) for j in selected) if selected else 0
            )
            if clip_negative:
                similarity = max(0, similarity)
            values.append(saliency[index] * (1 - similarity))
        selected.append(int(np.argmax(values)))
    result = np.zeros(len(features), dtype=bool)
    result[selected] = True
    return result


def test_independent_equations_include_negative_cosines():
    discriminating = 0
    for seed in range(32):
        rng = np.random.default_rng(seed)
        features, cls = (
            rng.normal(size=(12, 4)).astype(np.float32),
            rng.normal(size=4).astype(np.float32),
        )
        for count in (1, 3, 6, 12):
            expected = numpy_reference(features, cls, count)
            actual = select_current_frame(torch.from_numpy(features), torch.from_numpy(cls), count)
            np.testing.assert_array_equal(actual.numpy(), expected)
            if not np.array_equal(
                expected, numpy_reference(features, cls, count, clip_negative=True)
            ):
                discriminating += 1
    assert discriminating > 0, "The fixture must detect clipping away negative cosine similarity"


def test_zero_range_saliency_has_stable_unique_ties_not_diversity_fallback():
    features = torch.eye(4)
    cls = torch.zeros(4)
    assert select_current_frame(features, cls, 3).tolist() == [True, True, True, False]
    assert select_current_frame(torch.ones(4, 3), torch.ones(3), 2).tolist() == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("target", [0, -1, 5, True, 1.5])
def test_invalid_target(target):
    with pytest.raises(ValueError, match="target"):
        select_current_frame(torch.ones(4, 3), torch.ones(3), target)


def test_invalid_features():
    for features, cls in [
        (torch.ones(4, 3), torch.ones(2)),
        (torch.ones(4, 3, dtype=torch.long), torch.ones(3)),
        (torch.full((4, 3), float("nan")), torch.ones(3)),
        (torch.ones(4, 3), torch.full((3,), float("inf"))),
    ]:
        with pytest.raises(ValueError, match="finite aligned"):
            select_current_frame(features, cls, 2)


def setup_method(views=2, ratio=0.5):
    encoder = SimpleNamespace(
        training=False,
        cls_token=torch.ones(1, 1, 3),
        num_prefix_tokens=2,
        blocks=[torch.nn.Identity(), torch.nn.Identity()],
    )
    backend = SimpleNamespace(query_diagnostics={})
    method = CurrentFrameStrategy(backend, AMMRConfig(keep_ratio=ratio), encoder)
    size = 4
    layout = SimpleNamespace(
        views=views, patches_per_view=size, visual_mask=lambda p: (p >= 1) & (p < 1 + size * views)
    )
    state = SimpleNamespace(layout=layout, positions=torch.arange(views * size + 3))
    return method, encoder, state


def test_native_view_alignment_nonvisual_protection_and_lifetime():
    method, encoder, state = setup_method()
    method.before_query({})
    rng = np.random.default_rng(42)
    features = [torch.from_numpy(rng.normal(size=(1, 6, 3)).astype(np.float32)) for _ in range(2)]
    for feature in features:
        encoder.blocks[0](feature)
    method.start(state)
    keep = method.before_layer(0, state)
    assert keep[[0, 9, 10]].all()
    for view, feature in enumerate(features):
        expected = numpy_reference(feature[0, 2:].numpy(), feature[0, 0].numpy(), 2)
        np.testing.assert_array_equal(keep[1 + view * 4 : 5 + view * 4].numpy(), expected)
    assert method.probe.take() == []
    state.positions = state.positions[keep]
    assert method.before_layer(1, state) is None
    method.finish(state)
    assert method.backend.query_diagnostics["visual_tokens_final"] == 4
    assert method.layer_sizes == [7, 7]
    with pytest.raises(RuntimeError, match="twice"):
        method.before_layer(0, state)
    encoder.blocks[0](features[0])
    method.abort()
    assert method.probe.take() == []
    method.close()
    method.close()
    assert not encoder.blocks[0]._forward_hooks
    with pytest.raises(RuntimeError, match="closed"):
        method.before_query({})


def test_missing_or_wrong_geometry_rejected():
    method, encoder, state = setup_method()
    try:
        method.start(state)
        with pytest.raises(RuntimeError, match="one native DINO snapshot"):
            method.before_layer(0, state)
        for _ in range(2):
            encoder.blocks[0](torch.ones(1, 7, 3))
        with pytest.raises(ValueError, match="geometry"):
            method.before_layer(0, state)
    finally:
        method.close()


def test_encoder_requires_genuine_cls_and_eval():
    method, encoder, _ = setup_method()
    method.close()
    for kwargs in ({"cls_token": None}, {"training": True}, {"num_prefix_tokens": 0}):
        invalid = SimpleNamespace(**(vars(encoder) | kwargs))
        with pytest.raises(ValueError, match="genuine CLS"):
            CurrentFrameStrategy(SimpleNamespace(), AMMRConfig(), invalid)


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"keep_ratio": True},
        {"keep_ratio": 0},
        {"keep_ratio": 1.1},
        {"keep_ratio": float("nan")},
        {"keep_ratio": float("inf")},
        {"unknown": 1},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        AMMRConfig.parse(parameters)


@pytest.mark.gpu
@pytest.mark.parametrize("backend_id", ["openvla_oft", "cogact"])
def test_real_native_cls_alignment_selection_reset_rng_detach(
    backend_id, model_path, trace_path, request
):
    module = pytest.importorskip(f"efficient_vla_backend_{backend_id}.backend")
    from efficient_vla_benchmark.evaluation.traces import load_trace
    from efficient_vla_benchmark.model_ops.vision import VisionFeatureProbe

    implementation = importlib.import_module(
        f"efficient_vla_benchmark.methods.token_reduction.a_mmr.implementations.{backend_id}.method"
    ).AMMRCurrentFrame
    backend_cls = getattr(
        module, "CogACTBackend" if backend_id == "cogact" else "OpenVLAOFTBackend"
    )
    backend = backend_cls(model_path)
    request.addfinalizer(backend.close)
    if backend_id == "cogact":
        encoder = backend.model_access.vision.dino_featurizer
        decoder = backend.model_access.decoder
        views = 1
    else:
        vision = backend.model.vision_backbone
        encoder = next(
            x for x in (vision.featurizer, vision.fused_featurizer) if x.cls_token is not None
        )
        decoder = backend.model.language_model.model
        views = 2
    prefix = encoder.num_prefix_tokens
    probe = VisionFeatureProbe(encoder.blocks[-2])
    patches = []
    handle = encoder.register_forward_hook(
        lambda module, args, output: patches.append(output.detach().clone())
    )
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    try:
        native = backend.predict_chunk(observations[0])
        features = probe.take()
    finally:
        probe.close()
        handle.remove()
    assert len(features) == len(patches) == views
    for feature, patch in zip(features, patches):
        # Verifies selected block and norm convention against ACTUAL native output,
        # not just a guessed encoder layer or a synthetic CLS surrogate.
        torch.testing.assert_close(feature[:, prefix:], patch, rtol=0, atol=0)
    expected = [
        numpy_reference(f[0, prefix:].float().cpu().numpy(), f[0, 0].float().cpu().numpy(), 128)
        for f in features
    ]
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (AMMRConfig(enabled=False), AMMRConfig(keep_ratio=1.0)):
        method = implementation(backend, config)
        assert method.probe is None and method.adapter is None
        backend.attach_method(method)
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    # Exercise full native decoder instrumentation without selecting out any rows.
    dense = implementation(backend, AMMRConfig(keep_ratio=1.0))
    if backend_id == "cogact":
        dense.adapter = backend.model_access.attach_token_strategy(dense)
    else:
        dense.adapter = importlib.import_module(
            "efficient_vla_backend_openvla_oft.decoder"
        ).DecoderAdapter(decoder, dense)
    backend.attach_method(dense)
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(observations[0]), native, atol=1e-6, rtol=0)
    backend.detach_method()
    for _ in range(2):
        method = implementation(backend, AMMRConfig())
        selected = []

        def capture(module, args):
            selected.append(method.adapter.state.positions.cpu().numpy())

        handle = decoder.layers[0].register_forward_pre_hook(capture)
        backend.attach_method(method)
        backend.reset(context)
        try:
            actual = backend.predict_chunk(observations[0])
        finally:
            handle.remove()
        assert actual.shape == native.shape and np.isfinite(actual).all()
        assert torch.equal(native_rng, torch.cuda.get_rng_state())
        assert len(selected) == 1
        for view in range(views):
            positions = selected[0]
            ids = positions[(positions >= 1 + 256 * view) & (positions < 1 + 256 * (view + 1))]
            np.testing.assert_array_equal(ids, np.flatnonzero(expected[view]) + 1 + 256 * view)
        assert method.probe.take() == []
        assert backend.query_diagnostics["visual_tokens_final"] == views * 128
        assert len(set(backend.query_diagnostics["layer_sequence_lengths"])) == 1
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
        backend.detach_method()
        method.close()
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
