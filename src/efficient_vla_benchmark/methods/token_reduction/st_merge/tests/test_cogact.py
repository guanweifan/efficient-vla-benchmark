"""Real native paired-vision parity and spatial merging without action-head changes."""

from importlib import import_module

import numpy as np
import pytest
import torch

from ..config import STMergeConfig

pytest.importorskip("efficient_vla_backend_cogact")
STMerge = import_module(
    "efficient_vla_benchmark.methods.token_reduction.st_merge.implementations.cogact.method"
).STMerge


@pytest.mark.gpu
def test_real_cogact_pair_noop_merge_reset_rng_and_detach(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    vision = host.model_access.attach_vision_strategy(lambda *args: None)
    try:
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        assert vision.kept_positions.tolist() == list(range(1, 257))
        assert all(set(counts) == {256} for counts in vision.sequence_lengths.values())
    finally:
        vision.close()
    for config in (STMergeConfig(enabled=False), STMergeConfig(tokens_per_view=256)):
        host.attach_method(STMerge(host, config))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        host.detach_method()
    host.attach_method(STMerge(host, STMergeConfig()))
    host.reset(context)
    sparse = host.predict_chunk(frames[0])
    assert sparse.shape == (16, 7) and np.isfinite(sparse).all()
    assert 128 <= host.query_diagnostics["visual_tokens_final"] < 256
    for counts in host.query_diagnostics["vision_patch_counts"].values():
        assert counts[:2] == [256, 256] and counts[2] < 256
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), sparse)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
