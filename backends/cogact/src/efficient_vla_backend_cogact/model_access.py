"""Native component access for CogACT methods, never paper-specific decisions."""

from .decoder import CausalTokenAdapter, TokenLayout


class CogACTModelAccess:
    def __init__(self, model):
        self.decoder = model.vlm.llm_backbone.llm.model
        vision = model.vlm.vision_backbone
        self.vision = vision
        dino = vision.dino_featurizer.patch_embed.num_patches
        siglip = vision.siglip_featurizer.patch_embed.num_patches
        if dino != siglip or dino != 256:
            raise ValueError("This CogACT variant requires one aligned 16x16 patch grid")
        self.patches_per_view = dino

    def layout(self, length):
        return TokenLayout(length, patches_per_view=self.patches_per_view)

    def attach_token_strategy(self, strategy):
        return CausalTokenAdapter(self.decoder, strategy, self.layout)

    def policy_image(self, image):
        from .vision import policy_image

        return policy_image(self.vision, image)

    def attach_vision_strategy(self, callback):
        from .vision_adapter import PairedVisionAdapter

        return PairedVisionAdapter(self.vision, callback)

    def attach_reuse_strategy(self, strategy):
        from .reuse import CausalReuseAdapter

        return CausalReuseAdapter(self.decoder, strategy, self.layout)

    @property
    def attention_implementation(self):
        adapter = getattr(self.decoder, "_evb_token_adapter", None)
        strategy = adapter.strategy if adapter is not None else None
        return getattr(strategy, "attention_backend", self.decoder.config._attn_implementation)
