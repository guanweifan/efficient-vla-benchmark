"""Instance-local paired-ViT transforms with preserved native patch/readout layout.

Native timm forwards still execute each block. The adapter provides a shared
coordinate contract for both encoders; method callbacks own grouping decisions.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisionReduction:
    # One group per current patch; anchors index CURRENT patches in increasing order.
    groups: torch.Tensor
    anchors: torch.Tensor


def merge_patches(hidden, prefix, reduction, sizes):
    groups, anchors = reduction.groups, reduction.anchors
    patches = hidden[:, prefix:]
    count = anchors.numel()
    if (
        groups.dtype != torch.long
        or anchors.dtype != torch.long
        or groups.shape != (patches.shape[1],)
        or count < 1
        or count > patches.shape[1]
        or anchors.ndim != 1
        or groups.device != hidden.device
        or anchors.device != hidden.device
        or not bool(((groups >= 0) & (groups < count)).all())
        or not bool(((anchors >= 0) & (anchors < patches.shape[1])).all())
        or not bool((anchors[1:] > anchors[:-1]).all())
        or not torch.equal(groups[anchors], torch.arange(count, device=groups.device))
    ):
        raise ValueError("Invalid paired-vision grouping or anchor mapping")
    totals = sizes.new_zeros(count).index_add_(0, groups, sizes)
    if not bool((totals > 0).all()):
        raise ValueError("Empty visual merge group")
    merged = patches.new_zeros((patches.shape[0], count, patches.shape[-1]), dtype=torch.float32)
    merged.index_add_(1, groups, patches.float() * sizes[None, :, None])
    merged = (merged / totals[None, :, None]).to(hidden.dtype)
    return torch.cat((hidden[:, :prefix], merged), dim=1), totals


class PairedVisionTransform:
    """Run SigLIP first so a method can share its grouping with DINOv2.

    Restore sparse vision outputs into their original patch slots before the
    native projector. A consumer must remove unoccupied slots before LLM layer 0.
    This bridge preserves native proprio/action slicing; its dense projector cost
    is intentionally retained and timed. No checkpoint or global class is changed.
    """

    def __init__(self, backbone, callback, encoders):
        if getattr(backbone, "_evb_vision_adapter", None) is not None:
            raise ValueError("Only one paired-vision adapter may be attached")
        self.backbone, self.callback = backbone, callback
        if set(encoders) != {"dino", "siglip"}:
            raise ValueError("Paired vision requires exactly DINO and SigLIP encoders")
        self.encoders = encoders
        self.original_forward = backbone.forward
        self.handles = []
        self.kept_positions = None
        self.current_view = 0
        self.positions, self.sizes, self.source_groups = {}, {}, {}
        self.sequence_lengths = {}
        try:
            for branch, encoder in self.encoders.items():
                for index, block in enumerate(encoder.blocks):
                    self.handles.append(block.register_forward_hook(self._hook(branch, index)))
            backbone._evb_vision_adapter = self
            backbone.forward = self.forward
        except Exception:
            self.close()
            raise

    def _hook(self, branch, index):
        def hook(module, args, output):
            prefix = self.encoders[branch].num_prefix_tokens
            key = (self.current_view, branch)
            self.sequence_lengths[key].append(args[0].shape[1] - prefix)
            reduction = self.callback(
                self.current_view,
                branch,
                index,
                output,
                prefix,
                self.positions[branch],
                self.sizes[branch],
            )
            if reduction is not None:
                output, self.sizes[branch] = merge_patches(
                    output, prefix, reduction, self.sizes[branch]
                )
                self.positions[branch] = self.positions[branch][reduction.anchors]
                self.source_groups[branch] = reduction.groups[self.source_groups[branch]]
            return output

        return hook

    def forward(self, pixel_values):
        raise NotImplementedError("Backend binding owns native input unpacking")

    def forward_views(self, views):
        outputs, kept = [], []
        self.sequence_lengths = {}
        self.kept_positions = None
        for view, images in enumerate(views):
            self.current_view = view
            parts = {}
            for branch in ("siglip", "dino"):
                image = images[branch]
                if image.shape != (1, 3, 224, 224):
                    raise ValueError("Paired vision requires batch-one 224px branch images")
                self.positions[branch] = torch.arange(256, device=image.device)
                self.source_groups[branch] = self.positions[branch].clone()
                self.sizes[branch] = torch.ones(256, device=image.device)
                self.sequence_lengths[(view, branch)] = []
                sparse = self.encoders[branch](image)
                if sparse.shape[1] != self.positions[branch].numel():
                    raise ValueError("Visual transform must precede native penultimate readout")
                parts[branch] = sparse.new_zeros((1, 256, sparse.shape[-1])).index_copy_(
                    1, self.positions[branch], sparse
                )
            if not torch.equal(self.positions["dino"], self.positions["siglip"]):
                raise ValueError("DINO and SigLIP must share the same patch grouping anchors")
            if not torch.equal(self.source_groups["dino"], self.source_groups["siglip"]):
                raise ValueError("DINO and SigLIP must merge the same original patch members")
            outputs.append(torch.cat((parts["dino"], parts["siglip"]), dim=-1))
            kept.append(self.positions["siglip"] + 256 * view + 1)
        if not kept:
            raise ValueError("Paired vision requires at least one view")
        self.kept_positions = torch.cat(kept)
        return torch.cat(outputs, dim=1)

    def close(self):
        self.backbone.forward = self.original_forward
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.positions.clear()
        self.sizes.clear()
        self.source_groups.clear()
        self.kept_positions = None
        if getattr(self.backbone, "_evb_vision_adapter", None) is self:
            del self.backbone._evb_vision_adapter
