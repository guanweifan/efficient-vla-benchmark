import numpy as np


class MotionRouter:
    def __init__(self, threshold):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.use_low = False
        self.last_magnitude = None

    def observe(self, action, remaining):
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("MotionVQ needs finite native actions")
        if remaining == 0:
            self.last_magnitude = float(np.linalg.norm(action[:3]))
            # Public OFT code uses >=; the paper's prose uses >.
            self.use_low = self.last_magnitude >= self.threshold
