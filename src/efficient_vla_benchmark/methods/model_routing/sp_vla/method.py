"""Backend-independent scheduling in Cartesian delta-command space."""

from collections import deque

import numpy as np

from efficient_vla_benchmark.core.method import InferenceMethod


def extrapolate(history, ridge, translation_clip):
    """Pinned author's ridge fit: XYZ only, hold rotation and gripper."""
    actions = np.asarray(history, dtype=np.float64)
    x = np.column_stack((np.arange(len(actions)), np.ones(len(actions))))
    y = actions[:, :3]
    coefficients = np.linalg.solve(x.T @ x + ridge * np.eye(2), x.T @ y)
    prediction = np.clip(
        coefficients[0] * len(actions) + coefficients[1], -translation_clip, translation_clip
    )
    delta_max = np.max(np.abs(np.diff(y, axis=0)), axis=0)
    prediction = y[-1] + np.clip(prediction - y[-1], -delta_max, delta_max)
    return np.concatenate((prediction, actions[-1, 3:]))


class SPVLAScheduling(InferenceMethod):
    name = "sp_vla"
    requires_control_history = True

    def __init__(self, config):
        self.config = config
        self.reset(None)

    def reset(self, context):
        self.history = deque(maxlen=self.config.history_size)
        self.origins = deque(maxlen=self.config.history_size)
        self.generated = True
        self.replacements = 0

    def before_plan(self, observation, step):
        self.generated = True
        if (
            not self.config.enabled
            or not self.config.scheduling
            or len(self.history) < max(2, self.config.gate_steps)
            or sum(self.origins) < self.config.min_model_actions
        ):
            return None
        recent = np.asarray(list(self.history)[-self.config.gate_steps :])
        xy = np.maximum(np.max(np.abs(recent[:, :2]), axis=1), 0.0001)
        z = np.abs(recent[:, 2])
        if not np.all((z / xy < self.config.max_z_xy_ratio) & (z < self.config.max_z)):
            return None
        action = extrapolate(self.history, self.config.ridge, self.config.translation_clip)
        self.generated = False
        self.replacements += 1
        return action

    def on_plan(self, actions, step):
        self.generated = True

    def after_action(self, action, remaining):
        # Only executed commands enter history, never the unexecuted chunk tail.
        self.history.append(action.copy())
        self.origins.append(self.generated)
        return False

    def control_diagnostics(self):
        return {
            "scheduling_replacements": self.replacements,
            "model_actions_in_history": sum(self.origins),
        }

    def metadata(self):
        return {
            "scope": "SP-VLA scheduling core only; no token pruning",
            "reference_commit": "61872a4d6d8e9c18f08b6474ceca2ff2fe08e942",
            "fit_dimensions": "xyz; hold rotation and gripper",
            "adaptation": "schedule only at native chunk boundaries; execute one fitted action",
        }
