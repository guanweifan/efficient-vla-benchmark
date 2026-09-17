"""Semantic/action union, max-min filtering and causal decayed attention history."""

from collections import deque
from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod


def diverse_indices(features, count):
    if features.shape[0] <= count:
        return torch.arange(features.shape[0], device=features.device)
    normalized = features / features.norm(dim=1, keepdim=True).clamp_min(1e-6)
    distances = 1 - normalized @ normalized.T
    scores = distances.topk(2, dim=0, largest=False).values[1]
    selected = torch.empty(count, dtype=torch.long, device=features.device)
    available = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
    nearest = None
    for i in range(count):
        idx = scores.masked_fill(~available, -torch.inf).argmax()
        selected[i] = idx
        available[idx] = False
        row = distances.index_select(0, idx.reshape(1))[0]
        nearest = row if nearest is None else torch.minimum(nearest, row)
        scores = nearest
    return selected


def combine_filter(semantic, action, features, count):
    candidates = (
        torch.unique(torch.cat((semantic.topk(count).indices, action.topk(count).indices)))
        .sort()
        .values
    )
    if candidates.numel() > count:
        selected = diverse_indices(features.index_select(0, candidates), count)
        candidates = candidates.index_select(0, selected).sort().values
    return candidates


def temporal_scores(history, decay, dtype):
    weights = torch.tensor([decay**i for i in range(len(history))], device=history[0].device)
    score = torch.zeros_like(history[0], dtype=torch.float32)
    for w, previous in zip(weights, reversed(history)):
        score = score + w * previous.float()
    return (score / weights.sum().clamp_min(1e-8)).to(dtype)


class VLAPruner(InferenceMethod, DecoderStrategy):
    name = "vla_pruner"

    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.active = config.enabled and config.keep_ratio < 1
        self.requires_control_history = self.active and config.temporal
        self.reset(None)
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if self.active else None
        )

    def reset(self, context):
        self.history = deque(maxlen=self.config.history_size)
        self.layout = None
        self.queries = 0
        self.input_visual = None

    def before_query(self, prepared):
        self.queries += 1
        self.backend.query_diagnostics = {"query_count": self.queries}

    def start(self, state):
        if self.layout != state.layout:
            self.history.clear()
        self.layout = state.layout
        self.warm = self.config.temporal and len(self.history) < self.config.history_size
        self.previous_history = tuple(self.history)
        self.layer_sizes = []
        self.backend.query_diagnostics["warm_start"] = self.warm

    def before_layer(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        if layer == 0:
            self.input_visual = state.hidden[0, 1 : state.layout.visual_end].clone()

    def probes(self, layer, state):
        cfg, p, layout = self.config, state.positions, state.layout
        result = {}
        if layer == cfg.pruning_layer and not self.warm:
            result["semantic"] = p[p < layout.action_start]
            if not cfg.temporal:
                result["actions"] = p[(p >= layout.action_start) & (p < layout.length - 1)]
        if cfg.temporal and layer == cfg.history_layer:
            result["history"] = p[(p >= layout.action_start) & (p < layout.length - 1)]
        return result

    def after_layer(self, layer, state, attention):
        cfg, layout, p = self.config, state.layout, state.positions
        visual = layout.visual_mask(p)
        # Choose using OLD history, even if collection and pruning share a layer.
        old_history = self.previous_history
        if "history" in attention:
            full = attention["history"].new_zeros(layout.visual_end - 1)
            full[p[visual] - 1] = attention["history"][:, visual].mean(0)
            self.history.append(full.detach())
        if layer != cfg.pruning_layer or self.warm:
            return None
        semantic = attention["semantic"][:, visual].mean(0)
        action = (
            temporal_scores(old_history, cfg.decay, state.hidden.dtype)
            if cfg.temporal
            else attention["actions"][:, visual].mean(0)
        )
        count = max(1, round(layout.patches_per_view * cfg.keep_ratio))
        keep = ~visual
        for view in range(layout.views):
            start = view * layout.patches_per_view
            end = start + layout.patches_per_view
            ids = combine_filter(
                semantic[start:end], action[start:end], self.input_visual[start:end], count
            )
            keep[ids + start + 1] = True
        return keep

    def finish(self, state):
        self.input_visual = None
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.positions.numel() - (state.layout.length - 512),
            layer_sequence_lengths=self.layer_sizes,
            history_entries=len(self.history),
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
        self.history.clear()
        self.input_visual = None

    def metadata(self):
        return {
            "scope": "semantic/action combine-then-filter with decayed history",
            "source_commit": "84d4b7192c77abf1585610e2f12393319b7ebff9",
            "config": asdict(self.config),
            "requires_training": False,
            "attention_statistics": "FP32 head reduction on native bilateral attention",
        }
