# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0. See the root NOTICE.
# Modified: extracted the GR00T A3 decisions into a backend-independent controller;
# sampler callbacks return both execution commands and native normalized actions.
"""Official global confidence and dual verification, NOT the paper's medoid version."""

from dataclasses import dataclass, field

import numpy as np


def select_draft(actions, alpha, epsilon):
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[0] < 2 or actions.shape[2] != 7:
        raise ValueError("Expected multiple finite seven-dimensional action chunks")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite A3 candidates")
    u = np.cumsum(actions[:, :, :6], axis=1)
    mean, variance = np.mean(u, axis=0), np.var(u, axis=0)
    distance = np.sum((u - mean) ** 2 / (variance + float(epsilon)), axis=-1)
    confidence = np.exp(-0.5 * float(alpha) * distance)
    median = np.median(confidence, axis=0)
    high = confidence >= median[None, :]
    # Official code converts confidence to float32 BEFORE its weighted tie-break.
    confidence = confidence.astype(np.float32)
    count = np.sum(high, axis=1).astype(np.int32)
    total = np.sum(confidence * high, axis=1).astype(np.float32)
    candidates = np.flatnonzero(count == int(np.max(count)))
    best = int(candidates[np.argmax(total[candidates])])
    return best, confidence


@dataclass
class Branch:
    accepted: set[int] = field(default_factory=lambda: {0})
    active: bool = True
    fail_position: int | None = None
    fail_raw: np.ndarray | None = None
    stop_prefix: int | None = None
    rank: int = 0

    def prefix(self, horizon):
        for position in range(1, horizon):
            if position not in self.accepted:
                return position
        return horizon

    def verify(self, order, raw, commands, draft, thresholds, horizon):
        for row, position in enumerate(order):
            if np.all(np.abs(commands[row, position, :6] - draft[position, :6]) < thresholds):
                self.accepted.add(int(position))
                self.rank += 1
            else:
                self.fail_position = int(position)
                self.fail_raw = raw[row, position].copy()
                self.active = False
                self.stop_prefix = self.prefix(horizon)
                break


def decode(raw_candidates, commands, config, verify):
    """Return one native chunk, real execution prefix, and compact diagnostics.

    verify(fixed_actions, fixed_mask) samples a parallel batch conditioned on every
    earlier rank, including tentative ranks not yet accepted. Both branches are
    always enabled. Output beyond the returned prefix is storage, not execution.
    """
    raw_candidates, commands = np.asarray(raw_candidates), np.asarray(commands)
    if raw_candidates.shape != commands.shape or raw_candidates.ndim != 3:
        raise ValueError("Raw and execution candidates must have identical geometry")
    if not np.isfinite(raw_candidates).all():
        raise ValueError("Nonfinite normalized A3 candidates")
    k, native_horizon, dimension = raw_candidates.shape
    horizon = config.action_horizon
    if k != config.num_samples or horizon > native_horizon or dimension != 7:
        raise ValueError("A3 configured geometry differs from native samples")
    best, confidence = select_draft(
        commands[:, :horizon], config.confidence_alpha, config.confidence_epsilon
    )
    draft, raw_draft = commands[best, :horizon], raw_candidates[best]
    branches = [Branch(), Branch()]  # confidence, then temporal
    thresholds = np.asarray(config.delta_thresholds, dtype=np.float32) * config.threshold_scale
    verification_batches, verification_rows = 0, 0
    for start in range(0, horizon, config.verification_chunk_size):
        end = start + config.verification_chunk_size
        positions = np.arange(start, end, dtype=np.int64)
        orders = [
            positions[np.lexsort((positions, -confidence[best, positions]))],
            positions,
        ]
        orders = [
            np.asarray([p for p in order if p not in branch.accepted], dtype=np.int64)
            if branch.active
            else np.zeros(0, dtype=np.int64)
            for branch, order in zip(branches, orders)
        ]
        counts = [len(order) for order in orders]
        total = sum(counts)
        if not total:
            continue
        fixed = np.zeros((total, native_horizon, dimension), dtype=raw_candidates.dtype)
        masks = np.zeros((total, native_horizon), dtype=bool)
        offset = 0
        for branch, order in zip(branches, orders):
            known = sorted(p for p in branch.accepted if p < end)
            for row in range(len(order)):
                indices = known + order[:row].tolist()
                fixed[offset + row, indices] = raw_draft[indices]
                masks[offset + row, indices] = True
            offset += len(order)
        raw, execution = verify(fixed, masks)
        raw, execution = np.asarray(raw), np.asarray(execution)
        if raw.shape != fixed.shape or execution.shape != fixed.shape:
            raise ValueError("Verifier changed native batch geometry")
        if not np.isfinite(raw).all() or not np.isfinite(execution).all():
            raise ValueError("Nonfinite A3 verification actions")
        verification_batches += 1
        verification_rows += total
        offset = 0
        for branch, order in zip(branches, orders):
            if branch.active:
                stop = offset + len(order)
                branch.verify(
                    order, raw[offset:stop], execution[offset:stop], draft, thresholds, horizon
                )
            offset += len(order)
        conf, seq = branches
        if seq.active and conf.stop_prefix is not None and seq.prefix(horizon) >= conf.stop_prefix:
            seq.active = False
        if conf.active and seq.stop_prefix is not None and conf.prefix(horizon) >= seq.stop_prefix:
            conf.active = False
        if not conf.active and not seq.active:
            break

    conf, seq = branches
    prefixes = [branch.prefix(horizon) for branch in branches]
    prefix = min(prefixes)
    # On equal prefixes the official implementation prefers the temporal rejection.
    priority = (
        [conf]
        if prefixes[0] < prefixes[1]
        else ([seq] if prefixes[1] < prefixes[0] else [seq, conf])
    )
    selected = raw_draft.copy()
    append_position = -1
    executed = prefix
    for branch in priority:
        if branch.fail_position == prefix:
            selected[prefix] = branch.fail_raw
            executed += 1
            append_position = prefix
            break
    return (
        selected,
        executed,
        dict(
            selected_path=best,
            accepted_prefix_len=prefix,
            accepted_prefix_len_conf=prefixes[0],
            accepted_prefix_len_seq=prefixes[1],
            accepted_rank_conf=conf.rank,
            accepted_rank_seq=seq.rank,
            append_pos=append_position,
            verification_batches=verification_batches,
            verification_rows=verification_rows,
        ),
    )
