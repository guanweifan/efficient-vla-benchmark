"""One native RGB grid and prompt-only text rows for CogACT KV reuse."""

from efficient_vla_backend_cogact.reuse import ReuseStrategy

from ...method import InfoEntropyCore


class VLAInfoEntropy(InfoEntropyCore, ReuseStrategy):
    def __init__(self, backend, config):
        decoder = backend.model_access.decoder
        if max(config.reuse_layers) >= len(decoder.layers):
            raise ValueError("InfoEntropy reuse layer outside native decoder")
        super().__init__(backend, config, views=1, dtype=next(decoder.parameters()).dtype)
        if self.active:
            self.adapter = backend.model_access.attach_reuse_strategy(self)

    def before_query(self, prepared):
        images = [self.backend.model_access.policy_image(prepared["image"])] if self.active else []
        self.prepare_frame(prepared["instruction"], images)

    def text_bounds(self, layout):
        return layout.visual_end, layout.cognition_position

    def metadata(self):
        return {
            **super().metadata(),
            "camera_views": 1,
            "text_rows": "native prompt after visual prefix, excluding cognition EOS",
            "cognition": "always recomputed; native final-row readout and DDIM",
            "attention_backend": self.attention_backend,
            "attention_mask": "original causal query rows against all retained keys",
        }
