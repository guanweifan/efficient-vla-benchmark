import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import TemporalReuseConfig
from ..method import TemporalVisualReuse


def test_unchanged_input_reference_and_camera_isolation():
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(17)
    encoder = VisionTransformer(
        img_size=28, patch_size=14, embed_dim=24, depth=2, num_heads=3, num_classes=0
    ).eval()
    reference = copy.deepcopy(encoder)
    host = SimpleNamespace(query_diagnostics={})
    method = TemporalVisualReuse(host, TemporalReuseConfig(update_ratio=0.5), {"test": encoder}, 2)
    frames = [torch.randn(1, 3, 28, 28), torch.randn(1, 3, 28, 28)]
    with torch.inference_mode():
        for query in range(4):
            method.before_query({})
            for frame in frames:
                actual = encoder(frame)
                expected = reference(frame)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            method.after_query(None)
            assert host.query_diagnostics["vision_attention_reused_rows"] == (8 if query % 2 else 0)
        assert len(method.cache) == 4
        assert not torch.equal(method.cache["test", 0, 0][0], method.cache["test", 1, 0][0])
        assert all(0 in ids for ids in method.selected.values())
        method.reset(None)
        assert not method.cache
        method.close()
        method.close()
        assert "forward" not in encoder.__dict__


def test_config_noop_and_unsupported_batch():
    for bad in ({"update_ratio": 0}, {"update_ratio": True}, {"enabled": 1}, {"foo": 1}):
        with pytest.raises(ValueError):
            TemporalReuseConfig.parse(bad)
    from timm.models.vision_transformer import VisionTransformer

    encoder = VisionTransformer(
        img_size=28, patch_size=14, embed_dim=24, depth=2, num_heads=3, num_classes=0
    ).eval()
    host = SimpleNamespace(query_diagnostics={})
    method = TemporalVisualReuse(host, TemporalReuseConfig(update_ratio=1), {"x": encoder}, 1)
    assert not method.patches.saved
    method.close()
    method = TemporalVisualReuse(host, TemporalReuseConfig(), {"x": encoder}, 1)
    with torch.inference_mode():
        method.before_query({})
        with pytest.raises(ValueError, match="batch one"):
            encoder(torch.zeros(2, 3, 28, 28))
    assert not method.cache
    method.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend_name", ["cogact", "openvla_oft"])
def test_real_visual_native_reset_detach(backend_name, model_path, trace_path, request):
    pytest.importorskip("efficient_vla_backend_" + backend_name)
    if backend_name == "cogact":
        from efficient_vla_backend_cogact.backend import CogACTBackend as Backend

        from ..implementations.cogact.api import build
    else:
        from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend as Backend

        from ..implementations.openvla_oft.api import build
    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = Backend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    host.attach_method(build(host, TemporalReuseConfig(update_ratio=1), None))
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
    host.detach_method()
    host.attach_method(build(host, TemporalReuseConfig(), None))
    outputs = []
    for _ in range(2):
        host.reset(context)
        np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
        outputs.append(host.predict_chunk(frames[8]))
        assert np.isfinite(outputs[-1]).all()
        assert host.query_diagnostics["vision_attention_reused_rows"] > 0
        assert not host.query_diagnostics["vision_cache_refresh"]
    np.testing.assert_array_equal(*outputs)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
