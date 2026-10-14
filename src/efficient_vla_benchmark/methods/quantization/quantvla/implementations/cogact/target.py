"""Exact native CogACT language/DiT-MLP quantization boundary."""

from torch import nn


def targets(model):
    names = [
        f"vlm.llm_backbone.llm.model.layers.{i}.{suffix}"
        for i in range(32)
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    names += [
        f"action_model.net.blocks.{i}.mlp.{suffix}" for i in range(12) for suffix in ("fc1", "fc2")
    ]
    out = {name: model.get_submodule(name) for name in names}
    if any(not isinstance(m, nn.Linear) for m in out.values()):
        raise ValueError("QuantVLA requires unmodified native Linear targets")
    return out


def attentions(model):
    result = {
        f"action_model.net.blocks.{i}.attn": model.get_submodule(
            f"action_model.net.blocks.{i}.attn"
        )
        for i in range(12)
    }
    for module in result.values():
        if not all(
            hasattr(module, a)
            for a in ("qkv", "q_norm", "k_norm", "proj", "proj_drop", "fused_attn")
        ):
            raise ValueError("Unsupported CogACT attention topology")
    return result


def replace(model, name, module):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, module)
