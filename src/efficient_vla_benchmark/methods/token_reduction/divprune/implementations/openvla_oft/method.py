from dataclasses import asdict

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...selection import visual_selection


class DivPrune(InferenceMethod, DecoderStrategy):
    name = "divprune"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = None
        self.reset(None)
        if config.enabled and config.keep_ratio < 1:
            self.adapter = DecoderAdapter(backend.model.language_model.model, self)

    def reset(self, context):
        self.layer_sizes = []

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def before_layer(self, layer, state):
        if layer == 0:
            return visual_selection(
                state.hidden[0], state.layout.visual_mask(state.positions), self.config.keep_ratio
            )

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {}

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "max-min dispersion over projected visual embeddings before layer zero",
            "reference": "vbdi/divprune@799e2d950aa01ba7860907f5a6d86061f885dca6",
            "implementation_basis": "independent paper-based equations; no vendored source",
            "budget": "global across both cameras; nonvisual rows untouched",
            "numerics": "FP32 cosine distances; stable first-position ties; exclude selected IDs",
            "requires_training": False,
            "config": asdict(self.config),
        }
