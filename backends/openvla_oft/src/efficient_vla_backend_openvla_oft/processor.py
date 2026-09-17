"""Native OFT transforms and the explicit LIBERO action convention."""

from types import SimpleNamespace

import numpy as np
import torch


def prepare_observation(backend, observation):
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla

    if backend.context is None:
        raise RuntimeError("reset(context) must precede prediction")
    observation.validate()
    if observation.proprio.shape != (8,):
        raise ValueError("This OFT checkpoint requires eight-dimensional LIBERO proprioception")
    images = prepare_images_for_vla(
        [observation.image, observation.wrist_image], SimpleNamespace(center_crop=True)
    )
    prompt = f"In: What action should the robot take to {observation.instruction.lower()}?\nOut:"
    primary = backend.processor(prompt, images[0]).to(backend.device, dtype=backend.dtype)
    wrist = backend.processor(prompt, images[1]).to(backend.device, dtype=backend.dtype)
    primary["pixel_values"] = torch.cat([primary["pixel_values"], wrist["pixel_values"]], dim=1)
    proprio = normalize_proprio(
        observation.proprio.copy(), backend.model.norm_stats[backend.unnorm_key]["proprio"]
    )
    return {"inputs": primary, "proprio": proprio, "images": images, "observation": observation}


def process_actions(actions):
    actions = np.asarray(actions).copy()
    actions[:, -1] = -np.sign(2 * actions[:, -1] - 1)
    if actions.shape != (8, 7) or not np.isfinite(actions).all():
        raise ValueError("Invalid OFT policy action chunk")
    return actions
