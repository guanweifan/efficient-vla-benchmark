"""Targets supplied by a backend binding; projector and action modules remain native."""

from torch import nn
from torch.nn import functional as F


def targets(model, vision_prefix, language_prefix):
    found = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        side = (
            "vision"
            if name.startswith(vision_prefix)
            else "language"
            if name.startswith(language_prefix)
            else None
        )
        if side is None:
            continue
        # Native Prismatic uses intermediate patch/block features, bypassing
        # the optional timm pooling/classification heads.
        if side == "vision" and not any(part in name for part in (".blocks.", ".patch_embed.")):
            continue
        if isinstance(module, nn.Conv2d) and (
            module.groups != 1 or module.padding_mode != "zeros" or isinstance(module.padding, str)
        ):
            raise ValueError("Unsupported vision convolution")
        found[name] = (module, side)
    if not found or {side for _, side in found.values()} != {"vision", "language"}:
        raise ValueError("Both vision and language modules must be covered")
    return found


def flatten_input(module, inputs):
    if isinstance(module, nn.Conv2d):
        inputs = F.unfold(
            inputs, module.kernel_size, module.dilation, module.padding, module.stride
        ).transpose(1, 2)
    return inputs.reshape(-1, inputs.shape[-1])


def replace(model, name, module):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, module)
