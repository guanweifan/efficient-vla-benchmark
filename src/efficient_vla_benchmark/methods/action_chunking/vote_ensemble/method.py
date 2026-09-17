"""Paper equations 8-10 with explicit physical-time prediction alignment."""

from collections import deque

import numpy as np

from efficient_vla_benchmark.core.method import InferenceMethod


def ensemble_vote(candidates, threshold):
    candidates = np.asarray(candidates, dtype=np.float64)
    norms = np.linalg.norm(candidates, axis=1)
    denominator = norms * norms[-1]
    similarities = np.divide(
        candidates @ candidates[-1],
        denominator,
        out=np.zeros(len(candidates)),
        where=denominator > 0,
    )
    # Paper cosine is undefined at zero norm: identical zero vectors agree;
    # one zero and one nonzero vector have similarity zero.
    similarities[(norms == 0) & (norms[-1] == 0)] = 1
    similar = np.clip(similarities, -1, 1) > threshold
    similar[-1] = True  # A prediction always agrees with itself.
    majority = similar if int(similar.sum()) > int((~similar).sum()) else ~similar
    return candidates[majority].mean(axis=0), int(majority.sum())


class VoteEnsemble(InferenceMethod):
    name = "vote_ensemble"
    requires_control_history = True

    def __init__(self, config, encode, decode):
        self.config, self.encode, self.decode = config, encode, decode
        self.reset(None)

    def reset(self, context):
        self.predictions = deque(maxlen=self.config.max_history + 1)
        self.candidate_count = self.selected_count = 0

    def on_plan(self, actions, step):
        if self.config.enabled:
            self.predictions.append((step, self.encode(actions.copy())))

    def transform_action(self, action, step):
        if not self.config.enabled:
            return action
        while self.predictions and (
            step - self.predictions[0][0] > self.config.max_history
            or step >= self.predictions[0][0] + len(self.predictions[0][1])
        ):
            self.predictions.popleft()
        candidates = [
            chunk[step - start]
            for start, chunk in self.predictions
            if 0 <= step - start < len(chunk)
        ]
        if not candidates:
            raise RuntimeError("VOTE requires an aligned prediction for the current step")
        voted, self.selected_count = ensemble_vote(candidates, self.config.similarity_threshold)
        self.candidate_count = len(candidates)
        return self.decode(voted)

    def control_diagnostics(self):
        return {"vote_candidates": self.candidate_count, "vote_selected": self.selected_count}

    def metadata(self):
        return {
            "scope": "VOTE ensemble core only; no ACT-token training or action-head changes",
            "reference": "arXiv:2507.05116v4 equations 8-10",
            "implementation": "independent paper implementation; public TEV code not located",
            "zero_cosine": "both zero: 1; one zero: 0",
            "tie_rule": "average the not-similar group",
        }
