"""Native CogACT port contract; no OFT runtime dependency."""

import numpy as np
import pytest
import torch

from ..config import DARTConfig

pytest.importorskip("efficient_vla_backend_cogact")


@pytest.mark.gpu
def test_real_cogact_mechanism_reset_rng_and_detach(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import DART

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, observations, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(observations[0])
    native_rng = torch.cuda.get_rng_state().clone()
    host.attach_method(DART(host, DARTConfig(enabled=False)))
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
    host.detach_method()
    host.attach_method(DART(host, DARTConfig()))
    host.reset(context)
    actual = host.predict_chunk(observations[0])
    assert actual.shape == (16, 7) and np.isfinite(actual).all()
    count = host.query_diagnostics["visual_tokens_final"]
    expected_count = 132
    assert count == expected_count if expected_count is not None else 0 < count < 256
    lengths = host.query_diagnostics["layer_sequence_lengths"]
    assert len(lengths) == 32 and lengths[-1] < 300
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), actual)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
