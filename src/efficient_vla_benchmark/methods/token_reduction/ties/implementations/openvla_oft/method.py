from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...strategy import TIESStrategy


class TIES(TIESStrategy, DecoderStrategy):
    def __init__(self, backend, config, manifest=None, calibrating=False):
        super().__init__(backend, config, backend.model.language_model.model, manifest, calibrating)
        if self.active:
            try:
                self.adapter = DecoderAdapter(self.decoder, self)
                self.install_prefix()
            except Exception:
                self.close()
                raise

    def probes(self, layer, state):
        return {"ties": state.positions} if self.wants_scores(layer) else {}

    def after_layer(self, layer, state, attention):
        return self.consume_attention(layer, state, attention.get("ties"))
