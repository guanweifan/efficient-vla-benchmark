from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...reuse import CleanKVReuse


class ActuationSlackReuse(CleanKVReuse, DecoderStrategy):
    gate_description = "dense action-query attention; primary camera only; wrist always computed"

    def __init__(self, backend, config):
        decoder = backend.model.language_model.model
        super().__init__(backend, config, len(decoder.layers))
        if self.active:
            self.adapter = DecoderAdapter(decoder, self)

    def primary_mask(self, state):
        return state.layout.view_mask(state.positions, 0)

    def gate_queries(self, state):
        return state.positions[
            (state.positions >= state.layout.action_start)
            & (state.positions < state.layout.length - 1)
        ]

    def visual_count(self, layout):
        return layout.patches_per_view * layout.views
