"""Global set-level evolution; closed-loop fitness is supplied, never approximated."""

import math

import numpy as np


def validate_layout(layout, blocks, steps, budget):
    if len(layout) != blocks or not 2 * blocks <= budget <= blocks * steps:
        raise ValueError("Invalid global layout geometry or actual refresh budget")
    result = []
    for row in layout:
        if any(type(t) is not int or not 0 <= t < steps for t in row):
            raise ValueError("Invalid lattice timestep")
        if len(set(row)) != len(row) or not {0, steps - 1} <= set(row):
            raise ValueError("Every branch needs unique positions and counted endpoint refreshes")
        result.append(tuple(sorted(row)))
    if sum(map(len, result)) != budget:
        raise ValueError("Actual global refresh count differs from budget")
    return tuple(result)


def prior_weights(dissimilarity, epsilon=0.005, gamma=1.0):
    """Paper Eq. 7; min-max normalization matches the released importance code."""
    values = np.asarray(dissimilarity, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid activation dissimilarity prior")
    low, high = values.min(), values.max()
    normalized = (values - low) / (high - low) if high > low else np.full_like(values, 0.5)
    weights = (epsilon + normalized) ** gamma
    return weights / weights.sum()


class Evolution:
    """A compact bounded population, not a claim of exhaustive/converged search."""

    def __init__(self, prior, budget, seed, population_size=4, elite_size=2):
        self.prior = prior_weights(prior)
        self.blocks, self.steps = self.prior.shape
        if self.steps < 3 or not 2 * self.blocks < budget < self.blocks * self.steps:
            raise ValueError("Evolution requires an active, nontrivial cache budget")
        if not 3 <= population_size or not 1 <= elite_size < population_size:
            raise ValueError("Invalid evolutionary population")
        self.budget, self.population_size, self.elite_size = budget, population_size, elite_size
        self.rng = np.random.default_rng(seed)
        self.mandatory = frozenset(
            b * self.steps + t for b in range(self.blocks) for t in (0, self.steps - 1)
        )
        self.universe = tuple(i for i in range(self.blocks * self.steps) if i not in self.mandatory)
        self.free_budget = budget - len(self.mandatory)
        self.seen, self.fitness = set(), {}
        self.population = []

    def _repair(self, values):
        values = sorted(set(map(int, values)) - self.mandatory)
        if any(i not in self.universe for i in values):
            raise ValueError("Evolution received a foreign lattice position")
        if len(values) > self.free_budget:
            values = list(self.rng.choice(values, self.free_budget, replace=False))
        remaining = [i for i in self.universe if i not in values]
        values.extend(self.rng.choice(remaining, self.free_budget - len(values), replace=False))
        return tuple(sorted(map(int, [*values, *self.mandatory])))

    def _random(self, guided=False):
        weights = self.prior.flatten()[list(self.universe)] if guided else None
        if weights is not None:
            weights = weights / weights.sum()
        return self._repair(
            self.rng.choice(self.universe, self.free_budget, replace=False, p=weights)
        )

    def initial(self):
        if self.population:
            raise ValueError("Population already initialized")
        # Three guided + one random for the default population, preserving the
        # author's approximately 30% completely random initialization component.
        random_count = max(1, round(self.population_size * 0.3))
        for index in range(self.population_size):
            for _ in range(256):
                candidate = self._random(index >= random_count)
                if candidate not in self.seen:
                    break
            else:
                raise RuntimeError("Could not initialize unique schedules")
            self.population.append(candidate)
            self.seen.add(candidate)
        return list(self.population)

    def observe(self, candidate, successes, episodes):
        if candidate not in self.seen or candidate in self.fitness:
            raise ValueError("Foreign or already evaluated schedule")
        if (
            type(successes) is not int
            or type(episodes) is not int
            or not 0 <= successes <= episodes
            or episodes <= 0
        ):
            raise ValueError("Fitness requires complete counted rollout outcomes")
        self.fitness[candidate] = successes / episodes

    def next_generation(self):
        if any(c not in self.fitness for c in self.population):
            raise ValueError("Every current individual needs rollout fitness")
        ranked = sorted(self.population, key=lambda c: (-self.fitness[c], c))
        next_population, fresh = ranked[: self.elite_size], []

        def parent():
            indices = self.rng.choice(len(ranked), min(3, len(ranked)), replace=False)
            return min((ranked[i] for i in indices), key=lambda c: (-self.fitness[c], c))

        while len(next_population) < self.population_size:
            for _ in range(256):
                a, b = parent(), parent()
                first, second = (
                    [i for i in a if i not in self.mandatory],
                    [i for i in b if i not in self.mandatory],
                )
                self.rng.shuffle(first)
                self.rng.shuffle(second)
                values = (
                    [x if self.rng.random() < 0.5 else y for x, y in zip(first, second)]
                    if self.rng.random() < 0.8
                    else first
                )
                child = self._repair(values)
                # Author mutation: probability 0.1, replacing 2..6 positions
                # from the unrestricted mutable lattice, not the prior top-k.
                if self.rng.random() < 0.1:
                    old = [i for i in child if i not in self.mandatory]
                    remaining = [i for i in self.universe if i not in child]
                    limit = min(6, len(old), len(remaining))
                    count = int(self.rng.integers(2, limit + 1)) if limit >= 2 else 1
                    remove = set(self.rng.choice(old, count, replace=False))
                    child = self._repair(
                        [i for i in old if i not in remove]
                        + list(self.rng.choice(remaining, count, replace=False))
                    )
                if child not in self.seen:
                    break
            else:
                child = self._random()
                if child in self.seen:
                    raise RuntimeError("Search exhausted fresh schedules")
            self.seen.add(child)
            fresh.append(child)
            next_population.append(child)
        self.population = next_population
        return fresh

    def layout(self, candidate):
        rows = [
            [i % self.steps for i in candidate if i // self.steps == b] for b in range(self.blocks)
        ]
        return validate_layout(rows, self.blocks, self.steps, self.budget)


def confirm_target(quick, reference, confirmation=None, allowed_drop=0.03):
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (quick, reference)):
        raise ValueError("Invalid rollout fitness")
    trigger = quick >= reference
    return (
        trigger,
        trigger and confirmation is not None and confirmation >= reference - allowed_drop,
    )


class SearchBudget:
    """Reserve BEFORE launch; failed/interrupted allocations are never refunded."""

    def __init__(self, max_episodes=640, max_candidates=24):
        if (
            type(max_episodes) is not int
            or type(max_candidates) is not int
            or min(max_episodes, max_candidates) < 1
        ):
            raise ValueError("Search limits must be positive integers")
        self.max_episodes, self.max_candidates = max_episodes, max_candidates
        self.reservations = []

    def reserve(self, identity, kind, count):
        if (
            kind not in ("reference", "candidate", "confirmation")
            or type(count) is not int
            or count < 1
        ):
            raise ValueError("Invalid search allocation")
        if any(r["id"] == identity for r in self.reservations):
            raise ValueError("Search allocation already exists; no implicit retry")
        if sum(r["episodes"] for r in self.reservations) + count > self.max_episodes:
            raise ValueError("Authorized search episode budget would be exceeded")
        if (
            kind == "candidate"
            and sum(r["kind"] == kind for r in self.reservations) >= self.max_candidates
        ):
            raise ValueError("Authorized candidate limit would be exceeded")
        self.reservations.append(dict(id=identity, kind=kind, episodes=count))
