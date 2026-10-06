"""Full KV-reuse mechanism with image/text-attention entropy protection."""

import math
from dataclasses import asdict

import numpy as np
import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_statistics import AttentionStatisticsRequest
from efficient_vla_benchmark.model_ops.decoder import IndexedKVStore
from efficient_vla_benchmark.model_ops.vision import patch_cosine_similarity

from .entropy import image_entropy, quotas


def attention_information(mean_text_attention):
    words = mean_text_attention.shape[0]
    if words <= 1:
        return mean_text_attention.new_zeros(mean_text_attention.shape[-1])
    # Eq. 4 explicitly applies another softmax over text to AFTER-softmax attention.
    # FP64 entropy avoids cancellation for nearly uniform distributions; no
    # artificial temperature or rescaling to amplify the weak signal.
    q = mean_text_attention.double().softmax(0)
    entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log2()).sum(0)
    return (1 - entropy / math.log2(words)).clamp(0, 1)


def cache_schedule(entropies, growth):
    values = torch.stack(entropies)
    inverse = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
    result = inverse.tolist()
    for i in range(1, len(result)):
        if result[i] > result[i - 1]:
            result[i] = result[i - 1] + growth * (result[i] - result[i - 1])
    return result


class InfoEntropyCore(InferenceMethod):
    name = "vla_infoentropy"

    def __init__(self, backend, config, *, views, dtype):
        self.backend, self.config = backend, config
        self.views, self.dtype = views, dtype
        self.active = config.enabled and config.static_budget_per_view > 0
        self.requires_control_history = self.active
        self.attention_backend = config.attention_backend
        self.store = IndexedKVStore()
        self.reset(None)
        self.adapter = None

    def invalidate(self):
        self.store.clear()
        self.previous_images = self.previous_information = self.previous_schedule = None
        self.last_layout = None

    def reset(self, context):
        self.invalidate()
        self.last_instruction = None
        self.step = 0

    def before_plan(self, observation, step):
        self.step = step

    def after_action(self, action, remaining):
        self.step += 1
        return False

    def prepare_frame(self, instruction, images):
        self.backend.query_diagnostics = {}
        if not self.active:
            return
        if instruction != self.last_instruction:
            self.invalidate()
            self.step = 0
        self.last_instruction = instruction
        self.current_images = [np.array(image, copy=True) for image in images]
        if len(self.current_images) != self.views:
            raise ValueError("InfoEntropy image count must match the backend view count")
        visual_quota, attention_quota = quotas(
            self.step, self.config.low_quota, self.config.high_quota, self.config.transition_steps
        )
        candidates = []
        if self.previous_images is not None:
            for view in range(self.views):
                visual = image_entropy(self.current_images[view])
                informative = np.argsort(-visual, kind="stable")[:visual_quota]
                semantic = np.argsort(-self.previous_information[view], kind="stable")[
                    :attention_quota
                ]
                protected = np.union1d(informative, semantic)
                similarity = patch_cosine_similarity(
                    self.current_images[view], self.previous_images[view]
                )
                eligible = np.flatnonzero(similarity >= self.config.similarity_threshold)
                static = eligible[
                    np.argsort(-similarity[eligible], kind="stable")[
                        : self.config.static_budget_per_view
                    ]
                ]
                candidates.extend((np.setdiff1d(static, protected) + 1 + 256 * view).tolist())
        self.reusable = torch.tensor(candidates, device=self.backend.device, dtype=torch.long)
        self.backend.query_diagnostics.update(
            execution_step=self.step, visual_quota=visual_quota, attention_quota=attention_quota
        )

    def start(self, state):
        if self.last_layout != state.layout:
            self.invalidate()
            self.reusable = state.positions[:0]
        self.last_layout = state.layout
        self.cold = not self.store.keys
        self.full_positions = torch.arange(state.layout.length, device=state.positions.device)
        self.last_skipped = 0
        self.layer_sizes, self.entropies = [], []
        self.text_sum = None

    def before_layer(self, layer, state):
        selection = None
        if (
            not self.cold
            and self.previous_schedule is not None
            and self.reusable.numel()
            and layer in self.config.reuse_layers
        ):
            count = max(1, int(self.previous_schedule[layer] * self.reusable.numel()))
            if count > self.last_skipped:
                selection = ~torch.isin(state.positions, self.reusable[:count])
                self.last_skipped = count
        self.layer_sizes.append(state.layout.length - self.last_skipped)
        return selection

    def key_positions(self, layer, state):
        return self.full_positions

    def update_kv(self, layer, keys, values, state):
        return self.store.update(layer, keys, values, state.positions, state.layout.length)

    def probes(self, layer, state):
        text_start, text_end = self.text_bounds(state.layout)
        removed = state.layout.length - state.positions.numel()
        return {
            "all": AttentionStatisticsRequest(
                text_start - removed, text_end - removed, 1, state.layout.visual_end
            )
        }

    def after_layer(self, layer, state, attention):
        statistics = attention["all"]
        self.entropies.append(statistics.entropy)
        selected = statistics.text_attention
        self.text_sum = selected if self.text_sum is None else self.text_sum + selected

    def finish(self, state):
        self.previous_information = attention_information(self.text_sum / len(self.entropies))
        self.previous_information = self.previous_information.reshape(self.views, 256).cpu().numpy()
        self.previous_schedule = cache_schedule(self.entropies, self.config.growth_factor)
        self.text_sum = None
        self.backend.query_diagnostics.update(
            cache_cold=self.cold,
            reusable_candidates=self.reusable.numel(),
            tokens_per_layer=self.layer_sizes,
            reused_tokens_last_layer=self.last_skipped,
            next_reuse_proportions=self.previous_schedule,
            # All visual KV remain present; these are computed QUERY rows, not context size.
            visual_query_tokens_final=256 * self.views - self.last_skipped,
            visual_kv_tokens=256 * self.views,
        )

    def after_query(self, actions):
        if self.active:
            self.previous_images = self.current_images

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.invalidate()

    def abort(self):
        self.invalidate()
        self.text_sum = None
        self.entropies = []

    def text_bounds(self, layout):
        raise NotImplementedError

    def metadata(self):
        return {
            "scope": "entropy-guided protection, timestep allocation and full indexed KV reuse",
            "implementation_basis": "paper-based: arXiv:2604.05323v1; author code not verified",
            "config": asdict(self.config),
            "full_layer_entropy": True,
            "attention_source": "all-layer/head mean from PREVIOUS inference frame",
            "time_axis": "executed control steps, clamped at fixed T; not future episode length",
            "entropy_precision": "FP32 layer entropy; FP64 text-distribution entropy",
            "statistics": "compact complete-row entropy and complete text/visual rectangle",
            "attention_output": self.attention_backend,
            "requires_training": False,
        }
