"""Use CogACT's native DINO prefix; preserve causal cognition and DDIM sampling."""

from efficient_vla_backend_cogact.decoder import TokenStrategy

from ...strategy import CurrentFrameStrategy


class AMMRCurrentFrame(CurrentFrameStrategy, TokenStrategy):
    def __init__(self, backend, config):
        super().__init__(backend, config, backend.model_access.vision.dino_featurizer)
        try:
            if self.probe is not None:
                self.adapter = backend.model_access.attach_token_strategy(self)
        except Exception:
            self.close()
            raise
