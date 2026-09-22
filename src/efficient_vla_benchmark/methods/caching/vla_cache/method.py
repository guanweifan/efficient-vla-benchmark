"""Adaptive cross-frame indexed KV reuse with complete attention statistics."""

import numpy as np
import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.decoder import IndexedKVStore

from .config import VLACacheConfig
from .selection import reuse_schedule as reuse_schedule
from .selection import static_patch_ids as static_patch_ids


class VLACacheCore(InferenceMethod):
    name = "vla_cache"

    def __init__(self, backend, config: VLACacheConfig, *, views, dtype):
        self.backend, self.config = backend, config
        self.views, self.dtype = views, dtype
        self.attention_backend = config.attention_backend
        self.store = IndexedKVStore()
        self.adapter = None
        self.reset(None)

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

    def prepare_frame(self, instruction, images):
        if self.last_instruction != instruction:
            self._invalidate()
        self.last_instruction = instruction
        self.current_images = [np.array(image, copy=True) for image in images]
        candidates = []
        if self.previous_images is not None:
            for view in range(self.views):
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
            if count > self.last_skipped:
                keep = ~torch.isin(state.positions, self.reusable[:count])
                self.last_skipped = count
        # Candidates are unique visual positions and reuse removes a growing
        # prefix. Its exact size is known on CPU; do not synchronize a GPU sum.
        self.layer_sizes.append(state.layout.length - self.last_skipped)
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
        statistics_dtype = torch.float32 if self.dtype == torch.float16 else self.dtype
        probabilities = averaged.to(statistics_dtype)
        probabilities = probabilities / (probabilities.sum(dim=-1, keepdim=True) + 1e-10)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        entropy = -(probabilities * (probabilities + 1e-10).log()).sum(dim=-1).mean()
        self.entropies.append(entropy)
        if layer == self.config.task_attention_layer:
            text_start, text_end = self.text_bounds(state.layout)
            text = (state.positions >= text_start) & (state.positions < text_end)
            scores = averaged[text].mean(dim=0)
            self.next_protected = []
            for view in range(self.views):
                start = 1 + 256 * view
                order = torch.argsort(scores[start : start + 256], descending=True, stable=True)
                self.next_protected.append(order[: self.config.protected_tokens_per_view])
        return None

    def finish(self, state):
        self.last_positions = state.positions
        self.previous_schedule = reuse_schedule(
            self.entropies, self.config.weighted_growth, self.config.growth_factor
        )
        # Both views are consumed only by the next query. Avoid mid-decoder
        # device-to-host barriers, and transfer the indices together here.
        self.previous_protected = torch.stack(self.next_protected).tolist()
        self.backend.query_diagnostics = {
            "cache_cold": self.cold,
            "reusable_candidates": self.reusable.numel(),
            "tokens_per_layer": self.layer_sizes,
            "reused_tokens_last_layer": state.layout.length - state.positions.numel(),
            "next_reuse_proportions": self.previous_schedule,
            "visual_query_tokens_final": 256 * self.views - self.last_skipped,
            "visual_kv_tokens": 256 * self.views,
        }

    def after_query(self, actions):
        self.previous_images = self.current_images

    def metadata(self):
        return {
            "attention_backend": self.attention_backend,
            "full_layer_entropy": True,
            "statistics_queries": "all_active_rows",
        }

    def text_bounds(self, layout):
        raise NotImplementedError

    def abort(self):
        self._invalidate()

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self._invalidate()
