from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...selection import visual_selection


class DivPrune(InferenceMethod, TokenStrategy):
    name = "divprune"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = None
        if config.enabled and config.keep_ratio < 1:
            self.adapter = backend.model_access.attach_token_strategy(self)

    def before_query(self, prepared):
        self.backend.query_diagnostics = {}

    def before_layer(self, layer, state):
        if layer == 0:
            return visual_selection(
                state.hidden[0], state.layout.visual_mask(state.positions), self.config.keep_ratio
            )

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.layout.visual_count(len(state.positions)),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None

    def metadata(self):
        return {
            "scope": "max-min projected visual dispersion before native LLM layer zero",
            "reference": "vbdi/divprune@799e2d950aa01ba7860907f5a6d86061f885dca6",
            "implementation_basis": "shared independent paper-based equations",
            "numerics": "FP32 cosine distances; stable ties; unique selected IDs",
            "position_policy": "original causal positions and protected cognition",
            "requires_training": False,
            "config": asdict(self.config),
        }
