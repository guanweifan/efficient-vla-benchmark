from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe

from ...strategy import TIESStrategy


class TIES(TIESStrategy, TokenStrategy):
    def __init__(self, backend, config, manifest=None, calibrating=False):
        super().__init__(backend, config, backend.model_access.decoder, manifest, calibrating)
        if self.active:
            try:
                self.adapter = backend.model_access.attach_token_strategy(self)
                for layer, module in enumerate(self.decoder.layers):
                    self.native_probes.append(
                        NativeAttentionProbe(
                            module.self_attn,
                            lambda layer=layer: self.adapter.state.positions
                            if self.wants_scores(layer)
                            else None,
                            lambda: self.adapter.state.positions,
                        )
                    )
                self.install_prefix()
            except Exception:
                self.close()
                raise

    def after_layer(self, layer, state):
        probe = self.native_probes[layer]
        try:
            return self.consume_attention(layer, state, probe.probabilities)
        finally:
            probe.clear()
