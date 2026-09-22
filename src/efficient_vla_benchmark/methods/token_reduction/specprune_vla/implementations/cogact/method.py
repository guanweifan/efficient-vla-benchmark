"""Causal native-cognition saliency and real emitted-action control for SpecPrune."""

from dataclasses import asdict

import torch
from efficient_vla_backend_cogact.decoder import TokenStrategy
from PIL import Image

from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe

from ...method import SpecPruneCore


class SpecPrune(SpecPruneCore, TokenStrategy):
    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        decoder = backend.model_access.decoder
        if len(decoder.layers) != 32:
            raise ValueError("SpecPrune requires the native 32-layer CogACT backbone")
        self.attention_probes = {}
        self.requests = {}
        self.request_rows = {}
        super().__init__(backend, config, views=1)
        self.adapter = backend.model_access.attach_token_strategy(self)
        try:
            for layer in sorted({0, 1, *config.global_layers, *config.importance_layers}):
                self.attention_probes[layer] = NativeAttentionProbe(
                    decoder.layers[layer].self_attn,
                    lambda index=layer: self._request(index),
                    lambda: self.adapter.state.positions,
                )
        except Exception:
            self.close()
            raise

    def _request(self, layer):
        requests = self.probes(layer, self.adapter.state)
        self.requests[layer] = requests
        if requests:
            rows = torch.unique(torch.cat(list(requests.values())), sorted=True)
            self.request_rows[layer] = rows
            return rows

    def before_query(self, prepared):
        self.prepare_frame(prepared["instruction"], self.prepare_images([prepared["image"]]))

    def frame_images(self, observation):
        return (observation.image.copy(),)

    def prepare_images(self, images):
        return [
            self.backend.model_access.policy_image(
                image if isinstance(image, Image.Image) else Image.fromarray(image)
            )
            for image in images
        ]

    def text_bounds(self, layout):
        return layout.visual_end, layout.cognition_position

    def action_queries(self, state):
        return state.positions[-1:]

    def after_layer(self, layer, state):
        attention = {}
        probe = self.attention_probes.get(layer)
        if probe is not None and probe.probabilities is not None:
            for name, queries in self.requests.pop(layer).items():
                rows = torch.searchsorted(self.request_rows[layer], queries)
                attention[name] = probe.probabilities.index_select(0, rows)
            probe.clear()
            self.request_rows.pop(layer, None)
        return super().after_layer(layer, state, attention)

    def reset(self, context):
        super().reset(context)
        for probe in self.attention_probes.values():
            probe.clear()
        self.requests.clear()
        self.request_rows.clear()

    def close(self):
        for probe in self.attention_probes.values():
            probe.close()
        self.attention_probes.clear()
        super().close()

    def metadata(self):
        return {
            "scope": "global/local pruning, progressive importance and executed-action controller",
            "source_commit": "8091adc4b574ce9008d49a1dc9a210f4eec314c1",
            "query_adaptation": "native cognition condition for dynamic action-centric saliency",
            "global_local_attention": "text-to-visual, following pinned executable code",
            "source_parity": "shared author-code rules; independent CogACT binding",
            "camera_views": 1,
            "wrist_parameters": "unused on this single-view binding",
            "attention": "native causal SDPA; original positions and protected cognition",
            "diffusion_cache": "not composed with SpecPrune",
            "requires_training": False,
            "config": asdict(self.config),
        }
