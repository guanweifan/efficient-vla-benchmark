"""Actual native first-linear inputs, positions, reset and sampler preservation."""

import importlib

import numpy as np
import pytest
import torch

from ..config import VisionZipConfig
from .test_visionzip import numpy_evidence, numpy_reference


@pytest.mark.gpu
@pytest.mark.parametrize("backend_id", ["openvla_oft", "cogact"])
def test_real_native_preprojection_groups_positions_reset_rng_detach(
    backend_id, model_path, trace_path, request
):
    module = pytest.importorskip(f"efficient_vla_backend_{backend_id}.backend")
    from efficient_vla_benchmark.evaluation.traces import load_trace
    from efficient_vla_benchmark.model_ops.vision import VisionAttentionProbe

    implementation = importlib.import_module(
        f"efficient_vla_benchmark.methods.token_reduction.visionzip.implementations.{backend_id}.method"
    ).VisionZip
    backend_cls = getattr(
        module, "CogACTBackend" if backend_id == "cogact" else "OpenVLAOFTBackend"
    )
    backend = backend_cls(model_path)
    request.addfinalizer(backend.close)
    if backend_id == "cogact":
        encoder = backend.model_access.vision.siglip_featurizer
        decoder, projector, views = backend.model_access.decoder, backend.model.vlm.projector, 1
    else:
        vision = backend.model.vision_backbone
        encoder = next(
            x for x in (vision.featurizer, vision.fused_featurizer) if x.cls_token is None
        )
        decoder, projector, views = backend.model.language_model.model, backend.model.projector, 2
    probe = VisionAttentionProbe(encoder.blocks[-2].attn)
    fused = []
    handle = projector.register_forward_pre_hook(
        lambda module, args: fused.append(args[0].detach().clone())
    )
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    try:
        native = backend.predict_chunk(observations[0])
        qks = probe.take()
    finally:
        probe.close()
        handle.remove()
    assert len(fused) == 1 and len(qks) == views
    assert fused[0].shape[1] == 256 * views
    expected_features, expected_anchors = [], []
    for view, (query, key) in enumerate(qks):
        scores, metric = numpy_evidence(
            query.float().cpu().numpy(), key.float().cpu().numpy(), encoder.blocks[-2].attn.scale
        )
        features = fused[0][0, view * 256 : (view + 1) * 256].float().cpu().numpy()
        expected, anchors = numpy_reference(features, scores, metric, 128, 0.84375)
        expected_features.append(
            torch.tensor(expected, device=fused[0].device, dtype=fused[0].dtype)
        )
        expected_anchors.append(anchors + view * 256)
    compact = torch.cat(expected_features).unsqueeze(0)
    anchors = np.concatenate(expected_anchors)
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (VisionZipConfig(enabled=False), VisionZipConfig(keep_ratio=1)):
        method = implementation(backend, config)
        assert method.probe is None and method.adapter is None
        backend.attach_method(method)
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    dense = implementation(backend, VisionZipConfig(keep_ratio=1))
    if backend_id == "cogact":
        dense.adapter = backend.model_access.attach_token_strategy(dense)
    else:
        dense.adapter = importlib.import_module(
            "efficient_vla_backend_openvla_oft.decoder"
        ).DecoderAdapter(decoder, dense)
    backend.attach_method(dense)
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(observations[0]), native, rtol=0, atol=1e-6)
    backend.detach_method()
    linear = next(m for m in projector.modules() if isinstance(m, torch.nn.Linear))
    for _ in range(2):
        method = implementation(backend, VisionZipConfig())
        projected_inputs, selected = [], []
        linear_hook = linear.register_forward_pre_hook(
            lambda module, args: projected_inputs.append(args[0].detach().clone())
        )
        decoder_hook = decoder.layers[0].register_forward_pre_hook(
            lambda module, args: selected.append(method.adapter.state.positions.cpu().numpy())
        )
        backend.attach_method(method)
        backend.reset(context)
        try:
            actual = backend.predict_chunk(observations[0])
        finally:
            linear_hook.remove()
            decoder_hook.remove()
        assert actual.shape == native.shape and np.isfinite(actual).all()
        assert torch.equal(native_rng, torch.cuda.get_rng_state())
        assert len(projected_inputs) == len(selected) == 1
        assert projected_inputs[0].shape == compact.shape
        # Actual native fc1 input, not a merged-after-MLP proxy.
        torch.testing.assert_close(projected_inputs[0], compact, rtol=0, atol=0)
        visual_positions = selected[0][(selected[0] >= 1) & (selected[0] < 1 + views * 256)]
        np.testing.assert_array_equal(visual_positions, anchors + 1)
        assert backend.query_diagnostics["visionzip_projector_input_tokens"] == views * 128
        assert backend.query_diagnostics["visual_tokens_final"] == views * 128
        assert method.anchors is None and method.probe.take() == []
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
        backend.detach_method()
        method.close()
        assert not hasattr(projector, "_evb_projector_compression")
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
