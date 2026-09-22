"""Released SP-VLA pruning and scheduling on CogACT's native causal decoder."""

from dataclasses import asdict

import torch
from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.model_ops.vision import VisionAttentionProbe

from ...method import SPVLAScheduling
from ...pruning import attention_mass, edge_tokens, semantic_selection


class SPVLA(SPVLAScheduling, TokenStrategy):
    def __init__(self, backend, config):
        self.backend, self.probe, self.adapter = backend, None, None
        super().__init__(config)
        try:
            self.probe = VisionAttentionProbe(
                backend.model.vlm.vision_backbone.siglip_featurizer.blocks[-1].attn
            )
            self.adapter = backend.model_access.attach_token_strategy(self)
        except Exception:
            self.close()
            raise

    def reset(self, context):
        super().reset(context)
        if self.probe is not None:
            self.probe.clear()
        self.visual_keep = None

    def before_query(self, prepared):
        self.probe.clear()
        self.image = self.backend.model_access.policy_image(prepared["image"])
        z = abs(float(self.history[-1][2])) if self.history else 0.0
        self.mass = attention_mass(z, self.config.z_prune_threshold, self.config.max_prune_fraction)
        self.backend.query_diagnostics = {"attention_mass": self.mass, "executed_z": z}

    def start(self, state):
        pairs = self.probe.take()
        if len(pairs) != 1:
            raise RuntimeError("SP-VLA CogACT requires exactly one last-block SigLIP probe")
        q, k = pairs[0]
        semantic = semantic_selection(q, k, self.mass)
        edges = torch.as_tensor(
            edge_tokens(self.image, self.config.canny_low, self.config.canny_high),
            device=semantic.device,
        )
        if semantic.numel() != state.layout.patches_per_view:
            raise ValueError("SigLIP scores do not align with the fused patch grid")
        self.visual_keep = semantic | edges

    def before_layer(self, layer, state):
        if layer == 0:
            visual = state.layout.visual_mask(state.positions)
            keep = ~visual
            keep[visual] = self.visual_keep
            return keep

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(self.visual_keep.sum().item()),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
        )

    def abort(self):
        if self.probe is not None:
            self.probe.clear()

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        if self.probe is not None:
            self.probe.close()
            self.probe = None
        self.reset(None)

    def metadata(self):
        return {
            **super().metadata(),
            "scope": "SP-VLA visual pruning and optional same-paper model scheduling",
            "config": asdict(self.config),
            "pruning_rule": "released code: increasing absolute executed Z reduces attention mass",
            "position_policy": "aligned fused DINO/SigLIP patches; original causal LLM positions",
            "probe": "native last-block SigLIP normalized Q/K; second-last native vision output",
            "camera_views": 1,
            "extra_observation_inputs": [],
            "requires_training": False,
        }
