"""One-view paired vision merging followed by causal visual-row selection."""

from efficient_vla_backend_cogact.decoder import TokenStrategy

from ...method import STMergeCore


class STMerge(STMergeCore, TokenStrategy):
    def __init__(self, backend, config):
        super().__init__(backend, config)
        access = backend.model_access
        self.attach(
            access.vision.siglip_featurizer,
            access.attach_vision_strategy,
            access.attach_token_strategy,
        )

    def decoder_lengths(self):
        return list(self.adapter.layer_sequence_lengths)

    def metadata(self):
        return {
            **super().metadata(),
            "camera_views": 1,
            "attention": "native causal SDPA; original anchor positions and cognition EOS",
            "head": "unchanged native FP32 DDIM",
        }
