from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.key_probe import NativeKeyProbe

from ...selection import pivot_selection


class DART(InferenceMethod, TokenStrategy):
    name = "dart"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = self.probe = None
        if config.enabled and config.keep_ratio < 1:
            decoder = backend.model_access.decoder
            if config.pruning_layer >= len(decoder.layers):
                raise ValueError("Pruning must leave at least one decoder layer")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                self.probe = NativeKeyProbe(decoder.layers[config.pruning_layer - 1].self_attn)
            except Exception:
                self.close()
                raise

    def reset(self, context):
        self.abort()

    def abort(self):
        if self.probe:
            self.probe.clear()

    def before_query(self, prepared):
        self.abort()
        self.backend.query_diagnostics = {}

    def after_layer(self, layer, state):
        if layer != self.config.pruning_layer - 1:
            return
        cfg, layout = self.config, state.layout
        text = (state.positions >= layout.visual_end) & (
            state.positions < layout.cognition_position
        )
        try:
            hidden = self.backend.model_access.decoder.norm(state.hidden)[0]
            return pivot_selection(
                hidden,
                self.probe.keys,
                layout.visual_mask(state.positions),
                text,
                cfg.keep_ratio,
                cfg.visual_pivots,
                cfg.text_pivots,
            )
        finally:
            self.probe.clear()

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.layout.visual_count(len(state.positions)),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
        )
        self.abort()

    def close(self):
        if self.probe:
            self.probe.close()
            self.probe = None
        if self.adapter:
            self.adapter.close()
            self.adapter = None

    def metadata(self):
        return {
            "scope": "key-norm pivots and sequential visual diversity",
            "reference": "ZichenWen1/DART@0c888673198b0f2784053a5686f95a519ceef209",
            "pivots": "post-RoPE keys; text excludes native cognition EOS",
            "selection_features": "preceding output after native final RMSNorm",
            "budget": "per-pivot floor quota plus visual pivots (132/256 by default)",
            "ordering": "ascending original pivot positions; stable similarity ties",
            "requires_training": False,
            "config": asdict(self.config),
        }
