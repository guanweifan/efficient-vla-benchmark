from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.depth import depth_patch_signals
from efficient_vla_benchmark.model_ops.token_reduction import TokenReduction

from ...merging import ProgressiveDepthMerge


class DepthCachePrimary(InferenceMethod, DecoderStrategy):
    name = "depth_cache_primary"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.core = ProgressiveDepthMerge(config)
        self.adapter = None
        self.layer_sizes = []
        if config.enabled:
            decoder = backend.model.language_model.model
            if config.attention_layer >= len(decoder.layers):
                raise ValueError("Semantic attention layer outside decoder")
            self.adapter = DecoderAdapter(decoder, self)

    def reset(self, context):
        self.core.reset()
        self.layer_sizes = []

    def before_query(self, prepared):
        self.backend.query_diagnostics = {}
        self.layer_sizes = []
        if not self.config.enabled:
            return
        observation = prepared["observation"]
        if observation.depth is None:
            raise ValueError(
                "DepthCache primary core requires measured metric depth, not RGB estimates"
            )
        depth, edges = depth_patch_signals(
            observation.depth,
            self.backend.device,
            crop_area=0.9,
            edge_threshold=self.config.edge_threshold_m,
        )
        self.core.observe_depth(depth, edges)

    def before_layer(self, layer, state):
        if layer != 0:
            return
        primary = state.layout.view_mask(state.positions, 0)
        anchors, merged = self.core.reduce(state.hidden[0, primary])
        primary_ids = primary.nonzero().flatten()
        keep = ~primary
        keep[primary_ids[anchors]] = True
        # The second view remains completely native and full-resolution.
        wrist = state.layout.view_mask(state.positions, 1)
        features = torch.cat((merged, state.hidden[0, wrist]))
        return TokenReduction(keep, features)

    def probes(self, layer, state):
        self.layer_sizes.append(len(state.positions))
        if self.core.warming and layer == self.config.attention_layer:
            text = (state.positions >= state.layout.text_start) & (
                state.positions < state.layout.text_end
            )
            return {"semantic": state.positions[text]}
        return {}

    def after_layer(self, layer, state, attention):
        if self.core.warming and layer == self.config.attention_layer:
            self.core.collect_saliency(
                attention["semantic"][:, state.layout.view_mask(state.positions, 0)].mean(0)
            )

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=self.layer_sizes,
            **self.core.diagnostics(),
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "depth partition, gradual merging, dual protection and dynamic recovery",
            "exclusions": "carrying-dependent reinitialization; anticipatory wrist gating",
            "reference": "arXiv:2603.10469v1 III-B1/B2/B3",
            "inputs": "metric simulator/RGB-D depth; no object identities or grasp oracle",
            "warmup_clock": "five model-query frames, not extra policy calls",
            "semantic_signal": "native final-layer language-to-primary-visual attention",
            "geometry": "native resize and center-crop coordinates; nearest depth resampling",
            "thresholds": "explicit engineering defaults where publication does not specify values",
            "requires_training": False,
            "config": asdict(self.config),
        }
