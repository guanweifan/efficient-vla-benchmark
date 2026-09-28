"""Preserve native causal positions, nonlinear projector and diffusion readout."""

from efficient_vla_backend_cogact.decoder import TokenStrategy

from ...strategy import VisionZipStrategy


class VisionZip(VisionZipStrategy, TokenStrategy):
    def __init__(self, backend, config):
        super().__init__(
            backend,
            config,
            backend.model_access.vision.siglip_featurizer,
            backend.model.vlm.projector,
        )
        try:
            if self.probe is not None:
                self.adapter = backend.model_access.attach_token_strategy(self)
        except Exception:
            self.close()
            raise
