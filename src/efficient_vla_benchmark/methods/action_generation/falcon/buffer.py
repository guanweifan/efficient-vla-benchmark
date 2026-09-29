"""Bounded historical latents indexed by their actual PRE-update noise level."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LatentEntry:
    latent: object
    query: int
    step: int
    noise_level: int


class LatentBuffer:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Positive buffer capacity required")
        self.capacity = capacity
        self.entries = []

    def add(self, latent, query, step, noise_level):
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append(LatentEntry(latent.detach().clone(), query, step, noise_level))
        # Paper priority: oldest queries first, then highest noise first.
        self.entries.sort(key=lambda item: (item.query, -item.noise_level))

    def clear(self):
        self.entries.clear()
