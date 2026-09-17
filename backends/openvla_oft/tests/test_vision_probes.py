import torch
from efficient_vla_backend_openvla_oft.vision import VisionAttentionProbe
from timm.models.vision_transformer import Attention


@torch.inference_mode()
def test_probe_preserves_attention_and_drains_per_view_state():
    module = Attention(32, num_heads=4).eval()
    data = torch.randn(1, 16, 32)
    expected = module(data)
    probe = VisionAttentionProbe(module)
    torch.testing.assert_close(module(data), expected, rtol=0, atol=0)
    module(data)
    pairs = probe.take()
    assert len(pairs) == 2 and pairs[0][0].shape == (1, 4, 16, 8)
    assert probe.take() == []
    module(data)
    probe.close()
    assert not probe.keys and not module.q_norm._forward_hooks
