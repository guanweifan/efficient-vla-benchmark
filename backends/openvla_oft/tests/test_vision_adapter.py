import pytest
import torch
from efficient_vla_backend_openvla_oft.vision_adapter import VisionReduction, merge_patches


def test_size_weighted_merge_preserves_prefix_and_mass():
    x = torch.arange(18).reshape(1, 6, 3).float()
    groups = torch.tensor([0, 1, 0, 1])
    anchors = torch.tensor([0, 1])
    sizes = torch.tensor([1.0, 2.0, 3.0, 4.0])
    output, totals = merge_patches(x, 2, VisionReduction(groups, anchors), sizes)
    torch.testing.assert_close(output[:, :2], x[:, :2])
    torch.testing.assert_close(output[0, 2], (x[0, 2] + x[0, 4] * 3) / 4)
    torch.testing.assert_close(output[0, 3], (x[0, 3] * 2 + x[0, 5] * 4) / 6)
    assert totals.tolist() == [4, 6]


@pytest.mark.parametrize(
    "groups,anchors", [([1, 0], [0, 1]), ([0, 0], [0, 1]), ([0, 2], [0, 1]), ([0, 1], [1, 0])]
)
def test_reject_invalid_grouping(groups, anchors):
    with pytest.raises(ValueError):
        merge_patches(
            torch.ones(1, 2, 3),
            0,
            VisionReduction(torch.tensor(groups), torch.tensor(anchors)),
            torch.ones(2),
        )


def test_paired_adapter_rejects_different_members_with_identical_anchors():
    from efficient_vla_backend_openvla_oft.vision_adapter import PairedVisionAdapter
    from torch import nn

    class Encoder(nn.Module):
        num_prefix_tokens = 0

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        def forward(self, image):
            hidden = torch.arange(256).float().view(1, 256, 1)
            for block in self.blocks:
                hidden = block(hidden)
            return hidden

    class Backbone(nn.Module):
        use_fused_vision_backbone = True

        def __init__(self):
            super().__init__()
            self.featurizer, self.fused_featurizer = Encoder(), Encoder()

        def get_num_images_in_input(self):
            return 2

        def forward(self, image):
            return image

    def grouping(view, branch, layer, hidden, prefix, positions, sizes):
        if layer:
            return None
        anchors = torch.arange(0, 256, 2)
        groups = torch.arange(256) // 2
        if branch == "dino":
            groups[1] = 1
        return VisionReduction(groups, anchors)

    backbone = Backbone()
    adapter = PairedVisionAdapter(backbone, grouping)
    try:
        with pytest.raises(ValueError, match="same original patch members"):
            backbone(torch.zeros(1, 12, 224, 224))
    finally:
        adapter.close()
    assert not backbone.featurizer.blocks[0]._forward_hooks


@pytest.mark.gpu
def test_paired_real_vision_noop_is_exact_and_reversible(model_path, trace_path, request):
    import numpy as np
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend
    from efficient_vla_backend_openvla_oft.vision_adapter import PairedVisionAdapter

    from efficient_vla_benchmark.evaluation.traces import load_trace

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(observations[0])
    adapter = PairedVisionAdapter(backend.model.vision_backbone, lambda *args: None)
    try:
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), expected)
        assert adapter.kept_positions.tolist() == list(range(1, 513))
        assert all(set(counts) == {256} for counts in adapter.sequence_lengths.values())
    finally:
        adapter.close()
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), expected)
