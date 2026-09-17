"""Map LIBERO camera and state conventions into the shared observation type."""

import numpy as np

from efficient_vla_benchmark.core.types import Observation


def make_observation(raw: dict, instruction: str) -> Observation:
    quaternion = np.array(raw["robot0_eef_quat"], dtype=np.float64, copy=True)
    w = np.clip(quaternion[3], -1, 1)
    denominator = np.sqrt(1 - w * w)
    rotation = (
        np.zeros(3)
        if np.isclose(denominator, 0)
        else quaternion[:3] * 2 * np.arccos(w) / denominator
    )
    return Observation(
        raw["agentview_image"][::-1, ::-1].copy(),
        raw["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
        np.concatenate((raw["robot0_eef_pos"], rotation, raw["robot0_gripper_qpos"])),
        instruction,
    )
