from efficient_vla_backend_cogact.reuse import ReuseStrategy

from ...reuse import CleanKVReuse


class ActuationSlackReuse(CleanKVReuse, ReuseStrategy):
    gate_description = "dense native cognition-to-image attention; no DiT patch-attention claim"

    def __init__(self, backend, config):
        super().__init__(backend, config, len(backend.model_access.decoder.layers))
        if self.active:
            self.adapter = backend.model_access.attach_reuse_strategy(self)

    def primary_mask(self, state):
        return state.layout.visual_mask(state.positions)

    def gate_queries(self, state):
        return state.positions[-1:]

    def visual_count(self, layout):
        return layout.patches_per_view
