"""OFT two-camera packing for the shared native paired-vision transform."""

from efficient_vla_benchmark.model_ops.paired_vision import PairedVisionTransform
from efficient_vla_benchmark.model_ops.paired_vision import VisionReduction as VisionReduction
from efficient_vla_benchmark.model_ops.paired_vision import merge_patches as merge_patches


class PairedVisionAdapter(PairedVisionTransform):
    def __init__(self, backbone, callback):
        if not backbone.use_fused_vision_backbone or backbone.get_num_images_in_input() != 2:
            raise ValueError("Paired vision requires the two-view fused OFT backbone")
        super().__init__(
            backbone, callback, {"dino": backbone.featurizer, "siglip": backbone.fused_featurizer}
        )

    def forward(self, pixel_values):
        if pixel_values.shape != (1, 12, 224, 224):
            raise ValueError("Paired vision expects batch-one, two-camera OFT pixels")
        return self.forward_views(
            [
                {"dino": pixels[:, :3], "siglip": pixels[:, 3:]}
                for pixels in pixel_values.split(6, dim=1)
            ]
        )
