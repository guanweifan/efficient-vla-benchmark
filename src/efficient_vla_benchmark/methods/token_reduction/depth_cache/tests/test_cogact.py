import numpy as np
import pytest

from ..config import DepthCacheConfig

pytest.importorskip("efficient_vla_backend_cogact")


@pytest.mark.gpu
def test_real_cogact_depth_merge_reset_and_detach(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import DepthCachePrimary

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, observations, _ = load_trace(trace_path)
    assert observations[0].depth is not None
    host.reset(context)
    native = host.predict_chunk(observations[0])
    host.attach_method(DepthCachePrimary(host, DepthCacheConfig(enabled=False)))
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
    host.detach_method()
    host.attach_method(DepthCachePrimary(host, DepthCacheConfig()))
    host.reset(context)
    results, counts = [], []
    for _ in range(11):
        results.append(host.predict_chunk(observations[0]))
        counts.append(host.query_diagnostics["visual_tokens_final"])
    assert all(result.shape == (16, 7) and np.isfinite(result).all() for result in results)
    assert counts[:5] == [256] * 5 and 0 < counts[-1] < 256
    assert counts == sorted(counts, reverse=True)
    host.reset(context)
    for result in results:
        np.testing.assert_array_equal(host.predict_chunk(observations[0]), result)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
