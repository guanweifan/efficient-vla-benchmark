"""SP-VLA's released spatial/semantic selector and same-paper scheduling."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from efficient_vla_backend_openvla_oft.vision import VisionAttentionProbe

from ...method import SPVLAScheduling
from ...pruning import attention_mass as attention_mass
from ...pruning import edge_tokens as edge_tokens
from ...pruning import semantic_selection as semantic_selection


class SPVLA(SPVLAScheduling, DecoderStrategy):
    def __init__(self, backend, config):
        self.backend = backend
        super().__init__(config)
        self.probe = VisionAttentionProbe(
            backend.model.vision_backbone.fused_featurizer.blocks[-1].attn
        )
        self.adapter = DecoderAdapter(backend.model.language_model.model, self)

    def before_query(self, prepared):
        self.probe.clear()
        self.images = prepared["images"]
        z = abs(float(self.history[-1][2])) if self.history else 0.0
        self.mass = attention_mass(z, self.config.z_prune_threshold, self.config.max_prune_fraction)
        self.backend.query_diagnostics = {"attention_mass": self.mass, "executed_z": z}

    def start(self, state):
        pairs = self.probe.take()
        if len(pairs) != 2:
            raise RuntimeError("SP-VLA expected one SigLIP probe for each camera")
        masks = []
        for image, (q, k) in zip(self.images, pairs):
            semantic = semantic_selection(q, k, self.mass)
            edges = torch.as_tensor(
                edge_tokens(image, self.config.canny_low, self.config.canny_high),
                device=semantic.device,
            )
            masks.append(semantic | edges)
        self.visual_keep = torch.cat(masks)
        self.layer_sizes = []

    def before_layer(self, layer, state):
        if layer == 0:
            keep = ~state.layout.visual_mask(state.positions)
            keep[1 : state.layout.visual_end] = self.visual_keep
            return keep

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {}

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(self.visual_keep.sum().item()),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        self.adapter.close()
        self.probe.close()

    def metadata(self):
        return {
            **super().metadata(),
            "scope": "SP-VLA visual pruning and optional same-paper model scheduling",
            "config": asdict(self.config),
            "pruning_rule": "released code: increasing absolute executed Z reduces attention mass",
            "position_policy": "shared DINO/SigLIP patch mask; original LLM positions",
            "requires_training": False,
        }
