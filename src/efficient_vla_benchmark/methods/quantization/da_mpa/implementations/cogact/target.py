import re

from torch import nn


def targets(model):
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        language = re.fullmatch(
            r"vlm\.llm_backbone\.llm\.model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
            name,
        )
        dit = re.fullmatch(r"action_model\.net\.blocks\.(\d+)\.mlp\.(fc1|fc2)", name)
        if language:
            result[name] = (module, "language", None)
        elif dit:
            result[name] = (module, "dit_mlp", int(dit[1]))
    if (
        sum(s == "language" for _, s, _ in result.values()) != 224
        or sum(s == "dit_mlp" for _, s, _ in result.values()) != 24
    ):
        raise ValueError("Unsupported native DA-MPA target geometry")
    return result


def replace(model, name, module):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, module)
