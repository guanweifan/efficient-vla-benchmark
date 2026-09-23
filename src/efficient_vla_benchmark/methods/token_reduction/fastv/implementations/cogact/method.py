"""Released last-query FastV on native causal CogACT cognition EOS."""

from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe

from ...selection import attention_selection


class FastV(InferenceMethod, TokenStrategy):
    name = "fastv"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = self.probe = None
        if config.enabled and config.keep_ratio < 1:
            decoder = backend.model_access.decoder
            if config.pruning_layer >= len(decoder.layers):
                raise ValueError("Pruning must leave at least one decoder layer")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                self.probe = NativeAttentionProbe(
                    decoder.layers[config.pruning_layer - 1].self_attn,
                    lambda: self.adapter.state.positions[-1:],
                    lambda: self.adapter.state.positions,
                )
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
        if layer == self.config.pruning_layer - 1:
            try:
                return attention_selection(
                    self.probe.probabilities[0],
                    state.layout.visual_mask(state.positions),
                    self.config.keep_ratio,
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
            "scope": "released HF FastV last-query attention pruning",
            "reference": "pkunlp-icler/FastV@d1659729b5bf1be225e99ee15783deeea80f63b1",
            "query": "native final cognition EOS; no DiT action-to-patch substitution",
            "attention": "read-only native causal SDPA probe, normalized over all keys",
            "position_policy": "original RoPE and protected cognition; sampler unchanged",
            "requires_training": False,
            "config": asdict(self.config),
        }
