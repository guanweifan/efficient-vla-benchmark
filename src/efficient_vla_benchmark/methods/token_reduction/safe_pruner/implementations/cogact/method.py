"""Forecast native cognition-to-visual saliency, then refresh or prune."""

from dataclasses import asdict

import torch
from efficient_vla_backend_cogact.decoder import TokenStrategy
from torch.nn import functional as F

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe

from ...scoring import fused_selection, transfer_saliency


class SAFEPruner(InferenceMethod, TokenStrategy):
    name = "safe_pruner"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.active = config.enabled and config.keep_ratio < 1
        self.requires_control_history = self.active
        self.adapter = None
        self.probes = {}
        self.reset(None)
        if not self.active:
            return
        decoder = backend.model_access.decoder
        if max(config.pruning_layer, *config.reference_layers) >= len(decoder.layers):
            raise ValueError("SAFE-Pruner layer outside native decoder")
        self.adapter = backend.model_access.attach_token_strategy(self)
        try:
            for layer in sorted({*config.reference_layers, config.pruning_layer - 1}):
                self.probes[layer] = NativeAttentionProbe(
                    decoder.layers[layer].self_attn,
                    lambda index=layer: self._request(index),
                    lambda: self.adapter.state.positions,
                )
        except Exception:
            self.close()
            raise

    def _request(self, layer):
        if (
            layer == 0
            or (self.keyframe and layer in self.config.reference_layers)
            or (not self.keyframe and layer == self.config.pruning_layer - 1)
        ):
            return self.adapter.state.positions[-1:]

    def reset(self, context):
        self.reference_features = self.reference_saliency = self.layout = None
        self.queries = self.refreshes = 0
        self.abort()

    def abort(self):
        self.current_features = self.forecast = None
        self.new_saliency = {}
        for probe in self.probes.values():
            probe.clear()

    def before_query(self, prepared):
        self.queries += 1
        self.backend.query_diagnostics = {
            "query_count": self.queries,
            "visual_tokens_final": self.backend.model_access.patches_per_view,
        }

    def start(self, state):
        if self.layout != state.layout:
            self.reference_features = self.reference_saliency = None
        self.layout = state.layout
        self.keyframe = self.reference_features is None
        self.similarity = None
        self.new_saliency = {}

    def before_layer(self, layer, state):
        if layer == 0:
            self.current_features = state.hidden[
                0, state.layout.visual_mask(state.positions)
            ].clone()
            self.forecast = (
                None
                if self.reference_features is None
                else transfer_saliency(
                    self.current_features, self.reference_features, self.reference_saliency
                )
            )

    def after_layer(self, layer, state):
        cfg = self.config
        probe = self.probes.get(layer)
        if probe is None or probe.probabilities is None:
            return
        visual = state.layout.visual_mask(state.positions)
        saliency = probe.probabilities[0, visual]
        probe.clear()
        if layer == 0:
            if self.forecast is not None:
                self.similarity = float(
                    F.cosine_similarity(self.forecast[:1], saliency[None]).item()
                )
                self.keyframe = self.similarity < cfg.similarity_threshold
        if self.keyframe and layer in cfg.reference_layers:
            self.new_saliency[layer] = saliency.detach()
        if not self.keyframe and layer == cfg.pruning_layer - 1:
            count = max(1, round(state.layout.patches_per_view * cfg.keep_ratio))
            selected = fused_selection(saliency, self.forecast, cfg.forecast_weight, count)
            keep = ~visual
            keep[visual.nonzero().flatten()[selected]] = True
            return keep

    def finish(self, state):
        if self.keyframe:
            self.reference_features = self.current_features
            self.reference_saliency = torch.stack(
                [self.new_saliency[i] for i in self.config.reference_layers]
            )
            self.refreshes += 1
        self.backend.query_diagnostics.update(
            keyframe=self.keyframe,
            keyframe_similarity=self.similarity,
            refreshes=self.refreshes,
            visual_tokens_final=state.layout.visual_count(state.positions.numel()),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
        )
        self.abort()

    def close(self):
        for probe in self.probes.values():
            probe.close()
        self.probes.clear()
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "saliency transport, shallow/deep fusion and adaptive keyframe refresh",
            "implementation_basis": "SAFE-Pruner equations and published CogACT applicability",
            "query_adaptation": "native final cognition EOS, the sole VLM-to-DiT condition",
            "source_parity": "shared scorer; independent CogACT binding, not author-code parity",
            "preset": "fixed benchmark retention/gate; no LIBERO success tuning",
            "attention": "read-only native causal SDPA probes; all-key normalization",
            "requires_training": False,
            "config": asdict(self.config),
        }
