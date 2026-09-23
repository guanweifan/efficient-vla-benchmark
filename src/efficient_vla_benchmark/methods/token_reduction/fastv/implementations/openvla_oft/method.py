"""FastV with native OFT masks, positions and action readout."""

from dataclasses import asdict

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...selection import attention_selection


class FastV(InferenceMethod, DecoderStrategy):
    name = "fastv"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = None
        self.reset(None)
        if config.enabled and config.keep_ratio < 1:
            decoder = backend.model.language_model.model
            if config.pruning_layer >= len(decoder.layers):
                raise ValueError("Pruning must leave at least one decoder layer")
            self.adapter = DecoderAdapter(decoder, self)

    def reset(self, context):
        self.layer_sizes = []

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        if layer == self.config.pruning_layer - 1:
            return {"last_query": state.positions[-1:]}
        return {}

    def after_layer(self, layer, state, attention):
        if layer == self.config.pruning_layer - 1:
            return attention_selection(
                attention["last_query"][0],
                state.layout.visual_mask(state.positions),
                self.config.keep_ratio,
            )

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
            "scope": "released HF FastV last-query attention pruning",
            "reference": "pkunlp-icler/FastV@d1659729b5bf1be225e99ee15783deeea80f63b1",
            "query": "native final EOS, not fabricated action-to-image attention",
            "budget": "global across both visual views",
            "position_policy": "original positions; all nonvisual/readout tokens protected",
            "attention": "native bidirectional SDPA plus one selected-row probe",
            "requires_training": False,
            "config": asdict(self.config),
        }
