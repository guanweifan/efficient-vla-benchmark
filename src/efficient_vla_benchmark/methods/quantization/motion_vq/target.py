from torch import nn


def targets(model, prefix):
    found = {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and name.startswith(prefix)
    }
    if len(found) != 224 or any(
        m.in_features % 256 or m.out_features % 256 for m in found.values()
    ):
        raise ValueError("Unsupported native MotionVQ decoder")
    return found


def replace(model, name, module):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, module)
