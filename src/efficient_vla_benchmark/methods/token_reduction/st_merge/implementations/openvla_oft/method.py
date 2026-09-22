"""OFT paired-vision and action-slot boundary for shared spatial ST-Merge."""

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from efficient_vla_backend_openvla_oft.vision_adapter import PairedVisionAdapter

from ...method import STMergeCore
from ...method import spatial_groups as spatial_groups


class STMerge(STMergeCore, DecoderStrategy):
    def __init__(self, backend, config):
        super().__init__(backend, config)
        backbone = backend.model.vision_backbone
        self.attach(
            backbone.fused_featurizer,
            lambda callback: PairedVisionAdapter(backbone, callback),
            lambda strategy: DecoderAdapter(backend.model.language_model.model, strategy),
        )
