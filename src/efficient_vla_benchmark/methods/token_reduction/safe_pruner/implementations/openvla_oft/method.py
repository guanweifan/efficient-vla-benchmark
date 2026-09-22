"""Nearest-feature saliency transfer and adaptive full-frame refresh, without KV reuse."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from torch.nn import functional as F

from efficient_vla_benchmark.core.method import InferenceMethod

from ...scoring import correspondences as correspondences
from ...scoring import fused_selection, transfer_saliency


class SAFEPruner(InferenceMethod, DecoderStrategy):
    name = "safe_pruner"

    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.active = config.enabled and config.keep_ratio < 1
        self.requires_control_history = self.active
        self.reset(None)
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if self.active else None
        )

    def reset(self, context):
        self.reference_features = self.reference_saliency = None
        self.current_features = self.forecast = None
        self.layout = None
        self.queries = self.refreshes = 0

    def before_query(self, prepared):
        self.queries += 1
        self.backend.query_diagnostics = {"query_count": self.queries}

    def start(self, state):
        if self.layout != state.layout:
            self.reference_features = self.reference_saliency = None
        self.layout = state.layout
        self.keyframe = self.reference_features is None
        self.similarity = None
        self.layer_sizes = []
        self.new_saliency = {}

    def before_layer(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        if layer == 0:
            self.current_features = state.hidden[0, 1 : state.layout.visual_end].clone()
            self.forecast = (
                None
                if self.reference_features is None
                else transfer_saliency(
                    self.current_features, self.reference_features, self.reference_saliency
                )
            )

    def probes(self, layer, state):
        cfg, p, layout = self.config, state.positions, state.layout
        if layer == 0:
            return {"gate": p[-1:]}
        if (self.keyframe and layer in cfg.reference_layers) or (
            not self.keyframe and layer == cfg.pruning_layer - 1
        ):
            return {"action": p[(p >= layout.action_start) & (p < layout.length - 1)]}
        return {}

    def after_layer(self, layer, state, attention):
        cfg, layout = self.config, state.layout
        visual = layout.visual_mask(state.positions)
        if layer == 0:
            shallow = attention["gate"][0, visual]
            if self.forecast is not None:
                self.similarity = float(
                    F.cosine_similarity(self.forecast[:1], shallow.unsqueeze(0)).item()
                )
                self.keyframe = self.similarity < cfg.similarity_threshold
            if self.keyframe:
                self.new_saliency[0] = shallow.detach()
        elif self.keyframe and layer in cfg.reference_layers:
            self.new_saliency[layer] = attention["action"][:, visual].mean(0).detach()
        if layer == cfg.pruning_layer - 1 and not self.keyframe:
            shallow = attention["action"][:, visual].mean(0)
            count = max(1, round((layout.visual_end - 1) * cfg.keep_ratio))
            ids = fused_selection(shallow, self.forecast, cfg.forecast_weight, count)
            keep = ~visual
            keep[ids + 1] = True
            return keep

    def finish(self, state):
        if self.keyframe:
            self.reference_features = self.current_features
            self.reference_saliency = torch.stack(
                [self.new_saliency[i] for i in self.config.reference_layers]
            )
            self.refreshes += 1
        self.current_features = self.forecast = None
        self.new_saliency.clear()
        self.backend.query_diagnostics.update(
            keyframe=self.keyframe,
            keyframe_similarity=self.similarity,
            refreshes=self.refreshes,
            visual_tokens_final=state.positions.numel() - (state.layout.length - 512),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
        self.reset(None)

    def metadata(self):
        return {
            "scope": "feature-correspondence saliency forecast and adaptive keyframe refresh",
            "source_commit": "0094232dfdb332a087b246d56316a7c6c16c8598",
            "config": asdict(self.config),
            "requires_training": False,
            "keyframe_gate_query": "final token at layer zero, matching released OFT source",
            "forecast_queries": "final token at layer zero; action placeholders at deeper layers",
        }
