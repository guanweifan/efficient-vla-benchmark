from torch import nn


def targets(model):
    result = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
        and name.startswith(("vision_backbone.", "language_model."))
        and not name.startswith("language_model.lm_head")
    }
    if not result:
        raise ValueError("No native QVLA targets")
    for module in result.values():
        if isinstance(module, nn.Conv2d) and (module.groups != 1 or module.padding_mode != "zeros"):
            raise ValueError("Unsupported QVLA convolution")
    return result


def replace(model, name, module):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, module)
