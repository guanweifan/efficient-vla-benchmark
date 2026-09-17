"""SpecPrune-VLA mechanism adapted to the unmodified joint-suite OFT readout."""

import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from efficient_vla_backend_openvla_oft.vision import patch_cosine_similarity

from efficient_vla_benchmark.core.method import InferenceMethod

from ...config import SpecPruneConfig


def stable_patch_ids(current, previous, budget: int, threshold: float, offset: int):
    """Reference 14x14 RGB-patch cosine rule, in row-major visual-token order."""
    similarity = patch_cosine_similarity(current, previous)
    eligible = np.flatnonzero(similarity >= threshold)
    count = min(budget, len(eligible))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    selected = eligible[np.argpartition(similarity[eligible], -count)[-count:]]
    return selected.astype(np.int64) + offset


class SpecPrune(InferenceMethod, DecoderStrategy):
    name = "specprune_vla"

    def __init__(self, backend, config: SpecPruneConfig):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.requires_control_history = config.static_pruning or config.action_controller
        decoder = backend.model.language_model.model
        if len(decoder.layers) != 32:
            raise ValueError("This SpecPrune preset requires the 32-layer OFT backbone")
        self.reset(None)
        self.adapter = DecoderAdapter(decoder, self)

    def reset(self, context):
        self.frames = deque(maxlen=6)
        self.control_steps = 0
        self.query_count = 0
        self.replans = 0
        self.precise = False
        self.previous_mode = True
        self.action_sum = np.zeros(7, dtype=np.float64)
        self.previous_global = None
        self.confidence = {}
        self.last_layout = None
        self.last_instruction = None
        self.stable_ids = np.empty(0, dtype=np.int64)

    def observe(self, observation):
        self.control_steps += 1
        self.frames.append((observation.image.copy(), observation.wrist_image.copy()))

    def before_query(self, prepared):
        cfg = self.config
        if self.requires_control_history and not self.frames:
            raise RuntimeError(
                "SpecPrune requires observation callbacks from the common controller"
            )
        instruction = prepared["observation"].instruction
        if self.last_instruction != instruction:
            self.previous_global = None
            self.confidence.clear()
        self.last_instruction = instruction
        if cfg.static_pruning:
            from experiments.robot.openvla_utils import prepare_images_for_vla

            lag = max(
                1, min(6, int(-16 / 3 * np.linalg.norm(self.action_sum[:3]) / 6 + 22 / 3) + 3)
            )
            previous = self.frames[-lag] if len(self.frames) >= lag else self.frames[-1]
            previous = prepare_images_for_vla(list(previous), SimpleNamespace(center_crop=True))
            budgets = (
                (cfg.primary_precise_static_budget, cfg.wrist_precise_static_budget)
                if self.precise
                else (cfg.primary_static_budget, cfg.wrist_static_budget)
            )
            self.stable_ids = np.concatenate(
                [
                    stable_patch_ids(
                        prepared["images"][view],
                        previous[view],
                        budgets[view],
                        threshold,
                        1 + 256 * view,
                    )
                    for view, threshold in enumerate((cfg.primary_similarity, cfg.wrist_similarity))
                ]
            )
        self.action_sum[:] = 0
        self.query_count += 1
        self.backend.query_diagnostics = {
            "precise_mode": self.precise,
            "stable_patches": len(self.stable_ids),
            "control_step": self.control_steps,
            "replans": self.replans,
        }

    def after_action(self, action, remaining):
        cfg = self.config
        if not cfg.action_controller or self.control_steps < 10:
            return False
        self.action_sum += action
        if self.control_steps == 10:
            return False
        velocity = np.linalg.norm(action[:3])
        xy, z, rotation = np.linalg.norm(action[:2]), action[2], np.linalg.norm(action[3:6])
        replan = False
        if self.precise:
            if (
                z > cfg.exit_z_threshold or rotation > cfg.exit_rotation_threshold
            ) and remaining == 0:
                self.precise = False
        else:
            self.precise = bool(
                velocity < cfg.velocity_threshold
                or (
                    xy < cfg.xy_threshold
                    and z < cfg.z_threshold
                    and rotation < cfg.rotation_threshold
                )
            )
            replan = not self.previous_mode and self.precise and remaining > 0
        self.previous_mode = self.precise
        self.replans += int(replan)
        return replan

    def start(self, state):
        layout, device = state.layout, state.positions.device
        if self.last_layout != layout:
            self.previous_global = None
            self.confidence.clear()
        self.last_layout = layout
        self.scores = torch.zeros(layout.length, device=device)
        self.static_keep = torch.ones(layout.length, dtype=torch.bool, device=device)
        self.static_keep[torch.as_tensor(self.stable_ids, device=device)] = False
        if self.previous_global is not None:
            self.static_keep[self.previous_global] = True
        self.layer_zero_keep = torch.zeros_like(self.static_keep)
        self.next_global = []
        self.dynamic_active = self.config.dynamic_pruning
        self.layer_sizes = []
        self.updated_layers = []
        self.pruned_layers = []

    def before_layer(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return None

    def probes(self, layer, state):
        layout = state.layout
        device = state.positions.device
        probes = {}
        if self.config.static_pruning:
            if layer in (0, 1):
                skip = 9 if layer == 0 else 10
                start = min(layout.text_start + skip, layout.text_end - 1)
                probes["local_text"] = torch.arange(start, layout.text_end, device=device)
            if layer in self.config.global_layers:
                start = min(layout.text_start + 8, layout.text_end - 1)
                probes["global_text"] = torch.arange(start, layout.text_end, device=device)
        if self.dynamic_active and layer in self.config.importance_layers:
            probes["actions"] = torch.arange(layout.action_start, layout.length - 1, device=device)
        return probes

    def _local_indices(self, attention, state, view, count):
        mask = state.layout.view_mask(state.positions, view)
        positions = state.positions[mask]
        count = min(count, positions.numel())
        if not count:
            return positions[:0]
        scores = attention[:, mask].mean(dim=0)
        return positions[scores.topk(count).indices]

    def _collect_global(self, attention, state):
        for view, budget in enumerate(self.config.text_budgets(self.precise)):
            mask = state.layout.view_mask(state.positions, view)
            full = self.scores.new_zeros(256)
            full[state.positions[mask] - (1 + 256 * view)] = attention[:, mask].mean(dim=0)
            # The reference's Python stable sort breaks ties by original patch ID.
            selected = torch.argsort(full, descending=True, stable=True)[: budget // 2]
            self.next_global.append(selected + 1 + 256 * view)

    def _update_importance(self, layer, attention, state):
        visual = state.layout.visual_mask(state.positions)
        positions = state.positions[visual]
        if not positions.numel():
            return
        action_to_vision = attention[:, visual]
        mean_attention = action_to_vision.mean(dim=0)
        order = torch.sort(mean_attention, descending=True).indices
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.numel(), device=order.device)
        weights = torch.sigmoid(-ranks.float())
        weights = weights / (weights.sum() + 1e-8)
        if layer not in self.confidence:
            probabilities = action_to_vision + 1e-8
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
            entropy = entropy / math.log(max(positions.numel(), 2))
            self.confidence[layer] = (1 / (entropy + 1e-8)).detach()
        beta = self.config.importance_beta
        self.scores[positions] = (1 - beta) * self.scores[
            positions
        ] + beta * weights * self.confidence[layer]
        self.updated_layers.append(layer)

    def after_layer(self, layer, state, attention):
        cfg = self.config
        if "global_text" in attention:
            self._collect_global(attention["global_text"], state)
        if cfg.static_pruning and layer in (0, 1):
            keep = self.static_keep.clone()
            for view, budget in enumerate(cfg.text_budgets(self.precise)):
                selected = self._local_indices(
                    attention["local_text"], state, view, budget * (2 if layer == 0 else 1)
                )
                keep[selected] = True
                if layer == 0:
                    self.layer_zero_keep[selected[:budget]] = True
            if layer == 1:
                keep |= self.layer_zero_keep
            return keep[state.positions]
        if self.dynamic_active:
            if "actions" in attention:
                self._update_importance(layer, attention["actions"], state)
            if layer in cfg.pruning_layers:
                scores = self.scores[state.positions].clone()
                scores[~state.layout.visual_mask(state.positions)] = float("inf")
                protected = state.layout.length - 512
                count = min(
                    max(
                        int(state.positions.numel() * cfg.dynamic_keep_ratio),
                        protected + cfg.min_visual_tokens,
                    ),
                    state.positions.numel(),
                )
                if count == state.positions.numel():
                    self.dynamic_active = False
                keep = torch.zeros_like(state.positions, dtype=torch.bool)
                keep[torch.sort(scores, descending=True).indices[:count]] = True
                self.pruned_layers.append(layer)
                return keep
        return None

    def finish(self, state):
        if self.next_global:
            self.previous_global = torch.unique(torch.cat(self.next_global), sorted=True)
        self.backend.query_diagnostics.update(
            {
                "tokens_per_layer": self.layer_sizes,
                "final_visual_tokens": state.positions.numel() - (state.layout.length - 512),
                "importance_updates": self.updated_layers,
                "dynamic_pruning_layers": self.pruned_layers,
                "global_reuse_tokens": 0
                if self.previous_global is None
                else self.previous_global.numel(),
            }
        )

    def close(self):
        self.adapter.close()
