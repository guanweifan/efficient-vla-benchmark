from dataclasses import asdict

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...selection import pivot_selection


class DART(InferenceMethod, DecoderStrategy):
    name = "dart"

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
        self.keys = None
        self.layer_sizes = []

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def update_kv(self, layer, keys, values, state):
        if layer == self.config.pruning_layer - 1:
            self.keys = keys.transpose(1, 2).reshape(keys.shape[2], -1)
        return keys, values

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {}

    def after_layer(self, layer, state, attention):
        if layer != self.config.pruning_layer - 1:
            return
        if self.keys is None:
            raise RuntimeError("DART did not observe post-RoPE native keys")
        cfg, layout = self.config, state.layout
        text = (state.positions >= layout.text_start) & (state.positions < layout.text_end)
        try:
            # Released DART uses final decoder RMSNorm on the preceding layer output.
            hidden = self.backend.model.language_model.model.norm(state.hidden)[0]
            return pivot_selection(
                hidden,
                self.keys,
                layout.visual_mask(state.positions),
                text,
                cfg.keep_ratio,
                cfg.visual_pivots,
                cfg.text_pivots,
            )
        finally:
            self.keys = None

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=self.layer_sizes,
        )
        self.keys = None

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "Duplication-Aware Reduction of Tokens; not speculative decoding",
            "reference": "ZichenWen1/DART@0c888673198b0f2784053a5686f95a519ceef209",
            "pivots": "post-RoPE key L1 norms; language excludes proprio/action placeholders",
            "selection_features": "preceding layer output after native final RMSNorm",
            "budget": "floor(N*ratio/(visual_pivots+text_pivots)) per pivot plus visual pivots",
            "ordering": "ascending original pivot positions; stable similarity ties",
            "requires_training": False,
            "config": asdict(self.config),
        }
