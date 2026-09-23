import numpy as np
import pytest

from ..config import DepthCacheConfig


@pytest.mark.gpu
def test_real_oft_depth_merge_and_native_restore(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import DepthCachePrimary

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    assert observations[0].depth is not None, "Use measured schema-3 depth traces"
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    backend.attach_method(DepthCachePrimary(backend, DepthCacheConfig(enabled=False)))
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
    backend.detach_method()
    backend.attach_method(DepthCachePrimary(backend, DepthCacheConfig()))
    backend.reset(context)
    outputs, counts = [], []
    for _ in range(11):
        outputs.append(backend.predict_chunk(observations[0]))
        counts.append(backend.query_diagnostics["visual_tokens_final"])
    assert all(action.shape == (8, 7) and np.isfinite(action).all() for action in outputs)
    assert counts[:5] == [512] * 5 and 256 < counts[-1] < 512
    assert counts == sorted(counts, reverse=True)
    backend.reset(context)
    for output in outputs:
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), output)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
