"""Native no-CLS branch chooses per-camera fused groups before OFT projection."""

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...strategy import VisionZipStrategy


class VisionZip(VisionZipStrategy, DecoderStrategy):
    def __init__(self, backend, config):
        vision = backend.model.vision_backbone
        candidates = [
            x
            for x in (vision.featurizer, vision.fused_featurizer)
            if getattr(x, "cls_token", None) is None
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one native no-CLS SigLIP branch")
        super().__init__(backend, config, candidates[0], backend.model.projector)
        try:
            if self.probe is not None:
                self.adapter = DecoderAdapter(backend.model.language_model.model, self)
        except Exception:
            self.close()
            raise
