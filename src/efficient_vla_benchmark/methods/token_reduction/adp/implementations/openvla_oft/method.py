"""Training-free layer-zero Q/K selection and observed-motion gating."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...motion import MotionGate
from ...scoring import embedding_scores, view_quotas


class ADP(InferenceMethod, DecoderStrategy):
    name = "adp"

    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.active = config.enabled and config.keep_ratio < 1
        self.requires_control_history = self.active and config.dynamic
        self.reset(None)
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if self.active else None
        )

    def reset(self, context):
        self.gate = MotionGate(self.config)
        self.queries = self.pruned_queries = 0
        self.delta = 0.0
        self.should_prune = False

    def observe(self, observation):
        if self.requires_control_history:
            self.gate.observe(observation.proprio[:3])

    def before_query(self, prepared):
        if self.requires_control_history:
            if self.gate.last is None:
                raise RuntimeError("ADP requires the common controller's observation history")
            if self.queries:
                self.delta = self.gate.close_window()
        self.should_prune = self.active and (not self.config.dynamic or self.gate.state)
        self.queries += 1
        self.pruned_queries += int(self.should_prune)
        self.backend.query_diagnostics = {
            "pruned": self.should_prune,
            "observed_motion": self.delta,
            "query_count": self.queries,
            "pruned_queries": self.pruned_queries,
        }

    def start(self, state):
        self.layer_sizes = []

    def before_layer(self, layer, state):
        if layer != 0 or not self.should_prune:
            return None
        layout, h = state.layout, state.hidden
        text_ids = torch.cat(
            (
                state.positions[:1],
                state.positions[layout.text_start : layout.text_end],
                state.positions[-1:],
            )
        )
        scores = embedding_scores(
            self.backend.model.language_model.model.layers[0],
            h.index_select(1, text_ids),
            h[:, 1 : layout.visual_end],
        )
        quotas = view_quotas(
            round(scores.numel() * self.config.keep_ratio),
            self.config.view_weights,
            layout.patches_per_view,
        )
        keep = ~layout.visual_mask(state.positions)
        for view, quota in enumerate(quotas):
            start = view * layout.patches_per_view
            ids = scores[start : start + layout.patches_per_view].topk(quota).indices + start + 1
            keep[ids] = True
        return keep

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {}

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.positions.numel() - (state.layout.length - 512),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.adapter:
            self.adapter.close()

    def metadata(self):
        return {
            "scope": "embedding Q/K pruning with released observed-EEF motion gating",
            "source_commit": "d7094b09a4996847772c1fa975f09d863e1b759a",
            "scoring_layer": 0,
            "config": asdict(self.config),
            "position_policy": "preserve native original positions and readout",
            "requires_training": False,
        }
