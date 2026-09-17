"""Bounded-state port of the author's observed-EEF motion gate; no model imports."""

from collections import deque

import numpy as np


class MotionGate:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=config.window)
        self.state = False
        self.streak = 0
        self.average = None
        self.first = self.last = None
        self.distance = 0.0

    def observe(self, position):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("ADP requires a finite observed EEF position")
        if self.first is None:
            self.first = position.copy()
        if self.last is not None:
            self.distance += float(np.linalg.norm(position - self.last))
        self.last = position.copy()

    def close_window(self):
        if self.first is None:
            raise ValueError("ADP motion history is empty")
        cfg = self.config
        delta = (
            float(np.linalg.norm(self.last - self.first)) if cfg.delta == "net" else self.distance
        )
        if cfg.decision == "avg":
            self.average = delta if self.average is None else 0.5 * (self.average + delta)
            candidate = delta >= self.average
        else:
            self.history.append(delta)
            if len(self.history) < cfg.window:
                candidate = False
            else:
                previous = list(self.history)[:-1]
                if delta >= max(previous):
                    candidate = True
                elif delta < min(previous):
                    candidate = False
                else:
                    candidate = self.state if cfg.retain_previous_state else True
        if cfg.max_consecutive_pruned and candidate and self.streak >= cfg.max_consecutive_pruned:
            candidate = False
        self.streak = self.streak + 1 if candidate else 0
        self.state = bool(candidate)
        self.first = self.last.copy()
        self.distance = 0.0
        return delta
