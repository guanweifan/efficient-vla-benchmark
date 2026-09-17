"""Full KV-reuse mechanism with image/text-attention entropy protection."""

import math
from dataclasses import asdict

import numpy as np
import torch
from efficient_vla_backend_openvla_oft.decoder import (
    DecoderAdapter,
    DecoderStrategy,
    IndexedKVStore,
)
from efficient_vla_backend_openvla_oft.vision import patch_cosine_similarity

from efficient_vla_benchmark.core.method import InferenceMethod

from ...entropy import image_entropy, quotas


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


class VLAInfoEntropy(InferenceMethod, DecoderStrategy):
    name = "vla_infoentropy"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.active = config.enabled and config.static_budget_per_view > 0
        self.requires_control_history = self.active
        self.attention_backend = config.attention_backend
        self.store = IndexedKVStore()
        self.reset(None)
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if self.active else None
        )

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

    def before_query(self, prepared):
        self.backend.query_diagnostics = {}
        if not self.active:
            return
        instruction = prepared["observation"].instruction
        if instruction != self.last_instruction:
            self.invalidate()
            self.step = 0
        self.last_instruction = instruction
        self.current_images = [np.array(image, copy=True) for image in prepared["images"]]
        visual_quota, attention_quota = quotas(
            self.step, self.config.low_quota, self.config.high_quota, self.config.transition_steps
        )
        candidates = []
        if self.previous_images is not None:
            for view in range(2):
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
        return {"all": state.positions}

    def after_layer(self, layer, state, attention):
        averaged = attention["all"]
        dtype = torch.float32 if self.backend.dtype == torch.float16 else self.backend.dtype
        probability = averaged.to(dtype)
        probability = probability / (probability.sum(-1, keepdim=True) + 1e-10)
        probability = torch.nan_to_num(probability, nan=0.0)
        self.entropies.append(-(probability * (probability + 1e-10).log()).sum(-1).mean())
        text_mask = (state.positions >= state.layout.text_start) & (
            state.positions < state.layout.text_end
        )
        selected = averaged[text_mask, 1 : state.layout.visual_end]
        self.text_sum = selected if self.text_sum is None else self.text_sum + selected

    def finish(self, state):
        self.previous_information = attention_information(self.text_sum / len(self.entropies))
        self.previous_information = self.previous_information.reshape(2, 256).cpu().numpy()
        self.previous_schedule = cache_schedule(self.entropies, self.config.growth_factor)
        self.text_sum = None
        self.backend.query_diagnostics.update(
            cache_cold=self.cold,
            reusable_candidates=self.reusable.numel(),
            tokens_per_layer=self.layer_sizes,
            reused_tokens_last_layer=self.last_skipped,
            next_reuse_proportions=self.previous_schedule,
            # All visual KV remain present; these are computed QUERY rows, not context size.
            visual_query_tokens_final=512 - self.last_skipped,
            visual_kv_tokens=512,
        )

    def after_query(self, actions):
        if self.active:
            self.previous_images = self.current_images

    def close(self):
        if self.adapter:
            self.adapter.close()
        self.invalidate()

    def metadata(self):
        return {
            "scope": "entropy-guided protection, timestep allocation and full indexed KV reuse",
            "implementation_basis": "paper-based: arXiv:2604.05323v1; author code not verified",
            "config": asdict(self.config),
            "full_layer_entropy": True,
            "attention_source": "all-layer/head mean from PREVIOUS inference frame",
            "time_axis": "executed control steps, clamped at fixed T; not future episode length",
            "entropy_precision": "FP64 for text-distribution entropy to avoid cancellation",
            "requires_training": False,
        }
