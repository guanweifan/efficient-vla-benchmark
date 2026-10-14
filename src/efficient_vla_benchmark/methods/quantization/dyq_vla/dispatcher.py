"""Paper Eq. 4-6 and Algorithm 1, driven only by executed action history."""

from collections import deque

import numpy as np


class DyQDispatcher:
    def __init__(self, config, thresholds):
        self.config = config
        self.thresholds = tuple(
            float(thresholds[key]) for key in ("theta_2_4", "theta_4_8", "theta_fp")
        )
        if (
            not all(np.isfinite(self.thresholds))
            or not -1 <= self.thresholds[0] <= self.thresholds[1] <= self.thresholds[2]
        ):
            raise ValueError("Invalid DyQ threshold ordering")
        self.reset()

    def reset(self):
        self.motion = deque(maxlen=self.config.motion_window)
        self.jerk = deque(maxlen=self.config.jerk_window)
        self.motion_history = []
        self.jerk_history = []
        self.previous_rotation = None
        self.active_bits = 16
        self.counter = 0
        self.pending_max = 16
        self.last_sensitivity = 0.0
        self.last_target_bits = 16

    def observe(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("DyQ binding requires finite native xyz/rot/gripper commands")
        magnitude = float(np.linalg.norm(a[:3]))
        jerk = (
            0.0
            if self.previous_rotation is None
            else float(np.linalg.norm(a[3:6] - self.previous_rotation))
        )
        self.previous_rotation = a[3:6].copy()
        self.motion_history.append(magnitude)
        self.jerk_history.append(jerk)
        mu = max(float(np.percentile(self.motion_history, 95)), self.config.eta)
        nu = max(float(np.percentile(self.jerk_history, 95)), self.config.eta)
        # The paper clamps the fused state, NOT motion fineness separately.
        self.motion.append(1.0 - magnitude / mu)
        self.jerk.append(jerk / nu)
        self.last_sensitivity = max(
            0.0,
            self.config.sensitivity_lambda * float(np.mean(self.motion))
            + (1.0 - self.config.sensitivity_lambda) * float(np.mean(self.jerk)),
        )

    def target(self):
        s = self.last_sensitivity
        a, b, c = self.thresholds
        return 16 if not self.motion or s > c else (2 if s <= a else (4 if s <= b else 8))

    def dispatch_target(self, target):
        if type(target) is not int or target not in (2, 4, 8, 16):
            raise ValueError("Unsupported DyQ target precision")
        self.last_target_bits = target
        if target >= self.active_bits:
            self.active_bits, self.counter, self.pending_max = target, 0, target
        else:
            previous = self.pending_max
            self.pending_max = max(target, previous if self.counter > 0 else 0)
            self.counter = (self.counter if self.pending_max == previous else 0) + 1
            if self.counter == self.config.hysteresis_window:
                self.active_bits = self.pending_max
            self.counter %= self.config.hysteresis_window
        return self.active_bits

    def dispatch(self):
        return self.dispatch_target(self.target())

    def metadata(self):
        return dict(
            active_bits=self.active_bits,
            target_bits=self.last_target_bits,
            sensitivity=self.last_sensitivity,
            counter=self.counter,
        )
