"""The next deletion gate can ONLY originate in a complete dense refresh."""

import torch


class CleanGate:
    def __init__(self, keep_ratio):
        self.keep_ratio = keep_ratio
        self.reset()

    def reset(self):
        self.gate = None
        self.refreshing = False
        self.scores = []
        self.refreshes = 0

    def begin_refresh(self):
        if self.refreshing:
            raise RuntimeError("Nested dense refresh is unsupported")
        self.refreshing = True
        self.scores = []

    def record(self, scores):
        if not self.refreshing:
            raise RuntimeError("A served sparse forward cannot supply the next clean gate")
        self.scores.append(scores.detach().float())

    def publish(self):
        if not self.refreshing or not self.scores:
            raise RuntimeError("A complete dense refresh must supply saliency")
        self.gate = torch.stack(self.scores).mean(0)
        self.refreshes += 1
        self.scores = []

    def end_refresh(self):
        self.refreshing = False
        self.scores = []

    def selection(self, primary_mask):
        if self.refreshing or self.gate is None:
            return None
        ids = primary_mask.nonzero().flatten()
        if len(ids) != self.gate.numel():
            raise ValueError("ASR gate geometry changed without reset")
        budget = max(1, round(len(ids) * self.keep_ratio))
        chosen = torch.argsort(self.gate, descending=True, stable=True)[:budget]
        keep = ~primary_mask
        keep[ids[chosen]] = True
        return keep
