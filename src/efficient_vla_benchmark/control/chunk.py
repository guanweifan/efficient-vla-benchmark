"""Common action-chunk execution; retain the prefix, never the deque tail."""

from collections import deque

import numpy as np

from efficient_vla_benchmark.core.types import (
    ActionDecision,
    ChunkPredictor,
    EpisodeContext,
    Observation,
)


class ChunkPolicy:
    def __init__(self, predictor: ChunkPredictor, execution_horizon: int):
        if type(execution_horizon) is not int or execution_horizon < 1:
            raise ValueError("Execution horizon must be a positive integer")
        self.predictor = predictor
        self.execution_horizon = execution_horizon
        self._queue: deque[np.ndarray] = deque()
        self.query_count = 0
        self.executed_count = 0
        self._action_dim = None

    def reset(self, context: EpisodeContext) -> None:
        self._queue.clear()
        self.query_count = 0
        self.executed_count = 0
        self._action_dim = None
        self.predictor.reset(context)

    def step(self, observation: Observation) -> ActionDecision:
        self.predictor.observe(observation)
        queried = not self._queue
        if queried:
            actions = np.asarray(self.predictor.predict_chunk(observation))
            if (
                actions.ndim != 2
                or actions.shape[1] < 1
                or (self._action_dim is not None and actions.shape[1] != self._action_dim)
                or actions.shape[0] < self.execution_horizon
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    "Predictor must return enough finite actions with a consistent dimension"
                )
            self._action_dim = actions.shape[1]
            self._queue.extend(a.copy() for a in actions[: self.execution_horizon])
            self.query_count += 1
        self.executed_count += 1
        action = self._queue.popleft()
        replanned = self.predictor.after_action(action.copy(), len(self._queue))
        if replanned:
            self._queue.clear()
        return ActionDecision(action, queried, {"replan_requested": bool(replanned)})
