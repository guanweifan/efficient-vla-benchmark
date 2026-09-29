from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...strategy import ILSAStrategy


class ILSA(ILSAStrategy, DecoderStrategy):
    def __init__(self, backend, config):
        super().__init__(backend, config, backend.model.language_model.model)
        if self.active:
            self.adapter = DecoderAdapter(self.decoder, self)

    def language_mask(self, state):
        return (state.positions >= state.layout.text_start) & (
            state.positions < state.layout.text_end
        )

    def after_layer(self, layer, state, attention):
        return self.selection_after_layer(layer)
