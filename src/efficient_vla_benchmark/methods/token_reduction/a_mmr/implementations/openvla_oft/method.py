"""Use OFT's native CLS-bearing DINO branch, preserving paired patch membership."""

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...strategy import CurrentFrameStrategy


class AMMRCurrentFrame(CurrentFrameStrategy, DecoderStrategy):
    def __init__(self, backend, config):
        vision = backend.model.vision_backbone
        encoders = [vision.featurizer, vision.fused_featurizer]
        candidates = [
            encoder for encoder in encoders if getattr(encoder, "cls_token", None) is not None
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one native CLS-bearing DINO branch")
        super().__init__(backend, config, candidates[0])
        try:
            if self.probe is not None:
                self.adapter = DecoderAdapter(backend.model.language_model.model, self)
        except Exception:
            self.close()
            raise
