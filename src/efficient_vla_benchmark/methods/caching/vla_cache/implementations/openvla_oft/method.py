"""Adaptive cross-frame indexed KV reuse with complete attention statistics."""

import numpy as np
import torch
from efficient_vla_backend_openvla_oft.decoder import (
    DecoderAdapter,
    DecoderStrategy,
    IndexedKVStore,
)
from efficient_vla_backend_openvla_oft.vision import patch_cosine_similarity

from efficient_vla_benchmark.core.method import InferenceMethod

from ...config import VLACacheConfig


def static_patch_ids(current, previous, budget, threshold):
    similarity = patch_cosine_similarity(current, previous)
    # Match the reference's stable Python sort, not SpecPrune's argpartition.
    eligible = np.flatnonzero(similarity >= threshold)
    return eligible[np.argsort(-similarity[eligible], kind="stable")[:budget]]


def reuse_schedule(entropies, weighted_growth=True, growth_factor=0.55):
    entropies = torch.stack(entropies)
    normalized = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    reuse = 1 - normalized
    if weighted_growth:
        values = reuse.tolist()
        for index in range(1, len(values)):
            delta = values[index] - values[index - 1]
            if delta > 0:
                values[index] = values[index - 1] + delta * growth_factor
        return values
    return reuse.tolist()


class VLACache(InferenceMethod, DecoderStrategy):
    name = "vla_cache"

    def __init__(self, backend, config: VLACacheConfig):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.attention_backend = config.attention_backend
        decoder = backend.model.language_model.model
        if len(decoder.layers) != 32:
            raise ValueError("VLA-Cache presets require the 32-layer OFT backbone")
        self.store = IndexedKVStore()
        self.reset(None)
        self.adapter = DecoderAdapter(decoder, self)

    def _invalidate(self):
        self.store.clear()
        self.previous_images = None
        self.previous_protected = None
        self.previous_schedule = None
        self.reusable = None

    def reset(self, context):
        self._invalidate()
        self.last_layout = None
        self.last_instruction = None
        self.query_count = 0

    def before_query(self, prepared):
        instruction = prepared["observation"].instruction
        if self.last_instruction != instruction:
            self._invalidate()
        self.last_instruction = instruction
        self.current_images = [np.array(image, copy=True) for image in prepared["images"]]
        candidates = []
        if self.previous_images is not None:
            for view in range(2):
                stable = static_patch_ids(
                    self.current_images[view],
                    self.previous_images[view],
                    self.config.static_budget_per_view,
                    self.config.similarity_threshold,
                )
                remaining = sorted(set(stable.tolist()) - set(self.previous_protected[view]))
                candidates.extend(index + 1 + view * 256 for index in remaining)
        self.reusable = torch.as_tensor(candidates, dtype=torch.long, device=self.backend.device)
        self.query_count += 1

    def start(self, state):
        if self.last_layout != state.layout:
            self._invalidate()
            self.reusable = state.positions[:0]
        self.last_layout = state.layout
        self.full_positions = torch.arange(state.layout.length, device=state.positions.device)
        self.entropies = []
        self.next_protected = None
        self.last_skipped = 0
        self.layer_sizes = []
        self.cold = not self.store.keys

    def before_layer(self, layer, state):
        keep = None
        if (
            not self.cold
            and self.previous_schedule is not None
            and self.reusable.numel()
            and layer in self.config.reuse_layers
        ):
            count = max(1, int(self.previous_schedule[layer] * self.reusable.numel()))
            if count >= self.last_skipped:
                keep = ~torch.isin(state.positions, self.reusable[:count])
                self.last_skipped = count
        # Record final computed counts after this layer's incremental selection.
        self.layer_sizes.append(
            int(keep.sum().item()) if keep is not None else state.positions.numel()
        )
        return keep

    def key_positions(self, layer, state):
        return self.full_positions

    def update_kv(self, layer, keys, values, state):
        return self.store.update(layer, keys, values, state.positions, state.layout.length)

    def probes(self, layer, state):
        # Entropy uses every query row in every layer; sampling text rows alone
        # would change the algorithm's adaptive layer schedule.
        return {"all_queries": state.positions}

    def after_layer(self, layer, state, attention):
        averaged = attention["all_queries"]
        # The pinned entropy helper reduces and normalizes in the model dtype.
        statistics_dtype = (
            torch.float32 if self.backend.dtype == torch.float16 else self.backend.dtype
        )
        probabilities = averaged.to(statistics_dtype)
        probabilities = probabilities / (probabilities.sum(dim=-1, keepdim=True) + 1e-10)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        entropy = -(probabilities * (probabilities + 1e-10).log()).sum(dim=-1).mean()
        self.entropies.append(entropy)
        if layer == self.config.task_attention_layer:
            text = (state.positions >= state.layout.text_start) & (
                state.positions < state.layout.text_end
            )
            scores = averaged[text].mean(dim=0)
            self.next_protected = []
            for view in range(2):
                start = 1 + 256 * view
                order = torch.argsort(scores[start : start + 256], descending=True, stable=True)
                self.next_protected.append(order[: self.config.protected_tokens_per_view].tolist())
        return None

    def finish(self, state):
        self.previous_schedule = reuse_schedule(
            self.entropies, self.config.weighted_growth, self.config.growth_factor
        )
        self.previous_protected = self.next_protected
        self.backend.query_diagnostics = {
            "cache_cold": self.cold,
            "reusable_candidates": self.reusable.numel(),
            "tokens_per_layer": self.layer_sizes,
            "reused_tokens_last_layer": state.layout.length - state.positions.numel(),
            "next_reuse_proportions": self.previous_schedule,
        }

    def after_query(self, actions):
        self.previous_images = self.current_images

    def metadata(self):
        return {
            "attention_backend": self.attention_backend,
            "full_layer_entropy": True,
            "statistics_queries": "all_active_rows",
        }

    def close(self):
        self.adapter.close()
        self.store.clear()
