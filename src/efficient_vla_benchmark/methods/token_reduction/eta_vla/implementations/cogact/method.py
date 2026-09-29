from efficient_vla_backend_cogact.decoder import TokenStrategy

from ...strategy import ILSAStrategy


class ILSA(ILSAStrategy, TokenStrategy):
    def __init__(self, backend, config):
        super().__init__(backend, config, backend.model_access.decoder)
        if self.active:
            self.adapter = backend.model_access.attach_token_strategy(self)

    def language_mask(self, state):
        return (state.positions >= state.layout.visual_end) & (
            state.positions < state.layout.cognition_position
        )

    def after_layer(self, layer, state):
        return self.selection_after_layer(layer)
