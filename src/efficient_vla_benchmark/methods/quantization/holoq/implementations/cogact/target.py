"""CogACT-native scope: language projections and every DiT attention/MLP projection."""

from torch import nn


def targets(model):
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.startswith("vlm.llm_backbone.llm.model.layers.") and name.rsplit(".", 1)[-1] in {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }:
            result[name] = (module, "language")
        elif name.startswith("action_model.net.blocks."):
            if name.endswith((".attn.qkv", ".attn.proj")):
                result[name] = (module, "dit_attention")
            elif name.endswith((".mlp.fc1", ".mlp.fc2")):
                result[name] = (module, "dit_mlp")
    counts = {
        side: sum(kind == side for _, kind in result.values())
        for side in ("language", "dit_attention", "dit_mlp")
    }
    if counts != {"language": 224, "dit_attention": 24, "dit_mlp": 24}:
        raise ValueError("Unsupported or incomplete CogACT HoloQ target scope")
    return result


def replace(model, name, module):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, module)
