"""Explicit backend bindings supply their native decoder prefix."""

from torch import nn


def targets(model, prefix):
    result = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith(prefix)
        and name.rsplit(".", 1)[-1]
        in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    }
    if len(result) != 224:
        raise ValueError("DyQ requires complete native 224-projection decoder coverage")
    return result


def replace(model, name, op):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, op)
