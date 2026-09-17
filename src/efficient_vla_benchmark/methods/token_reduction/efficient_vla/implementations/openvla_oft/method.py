"""Paper section 3.3: key anchors, task augmentation and fixed-anchor diversity."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from torch.nn import functional as F

from efficient_vla_benchmark.core.method import InferenceMethod


def select_tokens(features, relevance, total, key_count, task_fraction):
    order = torch.argsort(relevance, descending=True, stable=True)
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    keys = order[:key_count]
    task_count = int(task_fraction * (total - key_count))
    selected[order[: key_count + task_count]] = True
    diversity_count = total - key_count - task_count
    if diversity_count:
        normalized = F.normalize(features.float(), dim=-1)
        # Eq. 4 compares against the ORIGINAL key set, not a growing FPS set.
        diversity = 1 - (normalized @ normalized[keys].T).amax(-1)
        diversity[selected] = -torch.inf
        selected[torch.argsort(diversity, descending=True, stable=True)[:diversity_count]] = True
    return selected


class EfficientVLA(InferenceMethod, DecoderStrategy):
    name = "efficient_vla"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.active = config.enabled and config.tokens_per_view < 256
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if self.active else None
        )

    def before_query(self, prepared):
        self.backend.query_diagnostics = {}

    def start(self, state):
        self.layer_sizes = []

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        if layer == self.config.pruning_layer:
            p, layout = state.positions, state.layout
            return {"text": p[(p >= layout.text_start) & (p < layout.text_end)]}
        return {}

    def after_layer(self, layer, state, attention):
        if layer != self.config.pruning_layer:
            return None
        layout, p = state.layout, state.positions
        scores = attention["text"].sum(0)
        keep = ~layout.visual_mask(p)
        for view in range(layout.views):
            mask = layout.view_mask(p, view)
            keep[mask] = select_tokens(
                state.hidden[0, mask],
                scores[mask],
                self.config.tokens_per_view,
                self.config.key_tokens_per_view,
                self.config.task_fraction,
            )
        return keep

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum().item()),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.adapter:
            self.adapter.close()

    def metadata(self):
        return {
            "scope": "token-pruning core only; no layer pruning or DiT caching",
            "implementation_basis": "paper-based: arXiv:2506.10100v1 section 3.3",
            "attention_direction": "text queries to visual keys",
            "feature_source": "post-block hidden states at pruning_layer",
            "config": asdict(self.config),
            "requires_training": False,
        }
