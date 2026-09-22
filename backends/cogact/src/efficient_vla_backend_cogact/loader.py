"""Offline assembly using pinned native classes and strict component loading.

Only construction adapts resource resolution: no upstream inference is rewritten,
no original weights are edited, and no unrelated pretrained weights are downloaded.
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import torch

_BUILD_LOCK = threading.Lock()


def load_native(root, checkpoint, statistics, device):
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import timm
    from prismatic.conf import ModelConfig
    from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
    from prismatic.models.backbones.llm.llama2 import LLaMa2LLMBackbone
    from prismatic.models.backbones.vision import dinosiglip_vit
    from prismatic.models.vlms.prismatic import PrismaticVLM
    from transformers import LlamaForCausalLM
    from vla import CogACT

    config = ModelConfig.get_choice_class("prism-dinosiglip-224px+7b")()
    state = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)["model"]
    components = {"vision_backbone", "llm_backbone", "projector", "action_model"}
    if set(state) != components:
        raise ValueError("Expected exactly the four complete native CogACT model components")

    class LocalLlamaBackbone(LLaMa2LLMBackbone):
        def __init__(self):
            HFCausalLLMBackbone.__init__(
                self,
                config.llm_backbone_id,
                "llama2",
                LlamaForCausalLM,
                str(root / "tokenizer"),
                llm_max_length=config.llm_max_length,
                inference_mode=True,
                use_flash_attention_2=False,
            )
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    def create_unloaded_vision(*args, **kwargs):
        kwargs["pretrained"] = False
        return timm.create_model(*args, **kwargs)

    # Replace only this constructor's module reference, not timm or any global class.
    # The context is serialized, restored on error, and gone before inference.
    factory = SimpleNamespace(create_model=create_unloaded_vision, data=timm.data)
    with _BUILD_LOCK, patch.object(dinosiglip_vit, "timm", factory):
        vision = dinosiglip_vit.DinoSigLIPViTBackbone(
            config.vision_backbone_id, config.image_resize_strategy
        )
    language = LocalLlamaBackbone()
    if language.llm.config._attn_implementation != "sdpa":
        raise ValueError("This baseline requires stock Transformers causal SDPA")
    vlm = PrismaticVLM(
        config.model_id,
        vision,
        language,
        arch_specifier=config.arch_specifier,
        enable_mixed_precision_training=True,
    )
    model = CogACT(
        vlm,
        action_model_type="DiT-B",
        action_dim=7,
        future_action_window_size=15,
        past_action_window_size=0,
        use_ema=False,
        norm_stats=statistics,
    )
    for name in ("vision_backbone", "llm_backbone", "projector"):
        getattr(vlm, name).load_state_dict(state[name], strict=True, assign=True)
    model.action_model.load_state_dict(state["action_model"], strict=True, assign=True)
    del state
    model.requires_grad_(False).eval()
    model.vlm.to(device=device, dtype=torch.bfloat16)
    # Native predict_action returns NumPy directly; retain its FP32 diffusion path.
    model.action_model.to(device=device, dtype=torch.float32)
    return model
