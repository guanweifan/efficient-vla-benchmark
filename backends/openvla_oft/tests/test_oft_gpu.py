from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.gpu
def test_native_prediction_parity_and_readonly_checkpoint(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend
    from efficient_vla_backend_openvla_oft.model_access import model_manifest

    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    from experiments.robot.openvla_utils import get_vla_action

    cfg = SimpleNamespace(
        num_images_in_input=2,
        center_crop=True,
        use_proprio=True,
        unnorm_key=backend.unnorm_key,
    )
    for sample in (samples[0], samples[len(samples) // 2], samples[-1]):
        original_proprio = sample.proprio.copy()
        actual = backend.predict_chunk(sample)
        reference = np.asarray(
            get_vla_action(
                cfg,
                backend.model,
                backend.processor,
                {
                    "full_image": sample.image,
                    "wrist_image": sample.wrist_image,
                    "state": sample.proprio.copy(),
                },
                sample.instruction,
                action_head=backend.action_head,
                proprio_projector=backend.proprio_projector,
            )
        )
        reference[:, -1] = -np.sign(2 * reference[:, -1] - 1)
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(sample.proprio, original_proprio)
    assert model_manifest(model_path) == backend._model_manifest


@pytest.mark.gpu
def test_instrumented_noop_matches_native_oft(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend
    from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    selected = (samples[0], samples[len(samples) // 2], samples[-1])
    expected = [backend.predict_chunk(sample) for sample in selected]
    adapter = DecoderAdapter(backend.model.language_model.model, DecoderStrategy())
    for sample, reference in zip(selected, expected):
        np.testing.assert_allclose(backend.predict_chunk(sample), reference, atol=1e-6, rtol=0)
    adapter.close()
    np.testing.assert_allclose(backend.predict_chunk(selected[0]), expected[0], atol=1e-6, rtol=0)
