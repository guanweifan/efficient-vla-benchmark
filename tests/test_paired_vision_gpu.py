"""An installed two-camera bridge must be native-equivalent before any grouping."""

import numpy as np
import pytest


@pytest.mark.gpu
def test_oft_active_identity_vision_bridge_preserves_both_views(model_path, trace_path, request):
    backend_module = pytest.importorskip("efficient_vla_backend_openvla_oft.backend")
    from efficient_vla_backend_openvla_oft.vision_adapter import PairedVisionAdapter

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = backend_module.OpenVLAOFTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    calls = set()

    def identity(view, branch, layer, hidden, prefix, positions, sizes):
        calls.add((view, branch))
        assert positions.tolist() == list(range(256))
        assert sizes.tolist() == [1] * 256
        return None

    adapter = PairedVisionAdapter(host.model.vision_backbone, identity)
    try:
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        assert calls == {(v, branch) for v in range(2) for branch in ("dino", "siglip")}
        assert adapter.kept_positions.tolist() == list(range(1, 513))
        assert all(set(counts) == {256} for counts in adapter.sequence_lengths.values())
    finally:
        adapter.close()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
