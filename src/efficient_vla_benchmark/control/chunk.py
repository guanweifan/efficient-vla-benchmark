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
        self._context = None
        self._instruction = None

    def reset(self, context: EpisodeContext) -> None:
        self._queue.clear()
        self.query_count = 0
        self.executed_count = 0
        self._action_dim = None
        self._context = context
        self._instruction = None
        self.predictor.reset(context)

    def step(self, observation: Observation) -> ActionDecision:
        if self._context is None:
            raise RuntimeError("reset(context) must precede control")
        if self._instruction is not None and observation.instruction != self._instruction:
            self._queue.clear()
            self._action_dim = None
            self.predictor.reset(self._context)
        self._instruction = observation.instruction
        self.predictor.observe(observation)
        step = self.executed_count
        queried = False
        action = None
        source = "queued"
        if not self._queue:
            hook = getattr(self.predictor, "before_plan", None)
            # Establish the native action contract with a real first query.
            action = hook(observation, step) if hook and self._action_dim is not None else None
            source = "replacement" if action is not None else "model"
        if not self._queue and action is None:
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
            hook = getattr(self.predictor, "on_plan", None)
            if hook:
                hook(actions.copy(), step)
            self._queue.extend(a.copy() for a in actions[: self.execution_horizon])
            self.query_count += 1
            queried = True
        if action is None:
            action = self._queue.popleft()
        action = self._validate_action(action)
        hook = getattr(self.predictor, "transform_action", None)
        if hook:
            action = self._validate_action(hook(action.copy(), step))
        replanned = self.predictor.after_action(action.copy(), len(self._queue))
        self.executed_count += 1
        if replanned:
            self._queue.clear()
        hook = getattr(self.predictor, "control_diagnostics", None)
        diagnostics = dict(hook()) if hook else {}
        diagnostics.update(replan_requested=bool(replanned), action_source=source)
        return ActionDecision(action, queried, diagnostics)

    def _validate_action(self, action):
        action = np.asarray(action)
        if (
            action.ndim != 1
            or action.size < 1
            or (self._action_dim is not None and action.size != self._action_dim)
            or not np.isfinite(action).all()
        ):
            raise ValueError("Control hook must return one finite action of consistent dimension")
        self._action_dim = action.size
        return action.copy()
