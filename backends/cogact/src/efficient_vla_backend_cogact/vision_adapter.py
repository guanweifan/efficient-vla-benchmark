"""Native CogACT dictionary inputs for paired DINO/SigLIP transformations."""

from efficient_vla_benchmark.model_ops.paired_vision import PairedVisionTransform


class PairedVisionAdapter(PairedVisionTransform):
    def __init__(self, backbone, callback):
        super().__init__(
            backbone,
            callback,
            {
                "dino": backbone.dino_featurizer,
                "siglip": backbone.siglip_featurizer,
            },
        )

    def forward(self, pixel_values):
        if not isinstance(pixel_values, dict) or set(pixel_values) != {"dino", "siglip"}:
            raise ValueError("CogACT paired vision requires its native dino/siglip dictionary")
        return self.forward_views([pixel_values])
