"""Native one-view paired geometry and shared group-membership invariants."""

import pytest
import torch
from efficient_vla_backend_cogact.vision_adapter import PairedVisionAdapter
from torch import nn

from efficient_vla_benchmark.model_ops.paired_vision import VisionReduction, merge_patches


class Encoder(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.num_prefix_tokens = prefix
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

    def forward(self, pixels):
        hidden = torch.arange(256 + self.num_prefix_tokens).float().view(1, -1, 1)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden[:, self.num_prefix_tokens :]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_featurizer = Encoder(5)
        self.siglip_featurizer = Encoder(0)

    def forward(self, pixels):
        return torch.cat(
            [self.dino_featurizer(pixels["dino"]), self.siglip_featurizer(pixels["siglip"])], -1
        )


def test_native_noop_dense_bridge_and_detach():
    backbone = Backbone()
    images = {name: torch.zeros(1, 3, 224, 224) for name in ("dino", "siglip")}
    expected = backbone(images)
    adapter = PairedVisionAdapter(backbone, lambda *args: None)
    torch.testing.assert_close(backbone(images), expected, rtol=0, atol=0)
    assert adapter.kept_positions.tolist() == list(range(1, 257))
    adapter.close()
    adapter.close()
    torch.testing.assert_close(backbone(images), expected, rtol=0, atol=0)
    assert not hasattr(backbone, "_evb_vision_adapter")


def test_identical_group_members_and_prefix_preservation():
    backbone = Backbone()
    images = {name: torch.zeros(1, 3, 224, 224) for name in ("dino", "siglip")}
    prefixes = []

    def group(view, branch, layer, hidden, prefix, positions, sizes):
        if layer != 0:
            if prefix:
                prefixes.append(hidden[:, :prefix].clone())
            return None
        return VisionReduction(torch.arange(256) // 2, torch.arange(0, 256, 2))

    adapter = PairedVisionAdapter(backbone, group)
    try:
        output = backbone(images)
        assert output.shape == (1, 256, 2)
        assert adapter.kept_positions.tolist() == list(range(1, 257, 2))
        torch.testing.assert_close(output[:, 1::2], torch.zeros_like(output[:, 1::2]))
        torch.testing.assert_close(output[0, 0], torch.tensor([5.5, 0.5]))
        for prefix in prefixes:
            torch.testing.assert_close(prefix.flatten(), torch.arange(5).float())
        assert all(counts == [256, 128, 128] for counts in adapter.sequence_lengths.values())
        with pytest.raises(ValueError, match="Only one"):
            PairedVisionAdapter(backbone, group)
    finally:
        adapter.close()


def test_weighted_mass_and_reject_mismatched_branches():
    x = torch.tensor([[[9.0], [1.0], [3.0], [5.0], [7.0]]])
    reduction = VisionReduction(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2]))
    out, mass = merge_patches(x, 1, reduction, torch.tensor([1.0, 3.0, 2.0, 2.0]))
    torch.testing.assert_close(out.flatten(), torch.tensor([9.0, 2.5, 6.0]))
    assert mass.tolist() == [4.0, 4.0]
    backbone = Backbone()

    def group(view, branch, layer, hidden, prefix, positions, sizes):
        if layer:
            return None
        groups = torch.arange(256) // 2
        if branch == "dino":
            groups[1] = 1
        return VisionReduction(groups, torch.arange(0, 256, 2))

    adapter = PairedVisionAdapter(backbone, group)
    try:
        with pytest.raises(ValueError, match="same original patch members"):
            backbone({name: torch.zeros(1, 3, 224, 224) for name in ("dino", "siglip")})
    finally:
        adapter.close()
