"""Compress native fused features before sparse pointwise projection and LLM prefill."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches
from efficient_vla_benchmark.model_ops.vision import VisionAttentionProbe

from .grouping import compress_view, visual_evidence


class VisionZipStrategy(InferenceMethod):
    name = "visionzip"

    def __init__(self, backend, config, encoder, projector):
        if (
            encoder.training
            or projector.training
            or getattr(encoder, "cls_token", None) is not None
            or getattr(encoder, "num_prefix_tokens", 0) != 0
            or len(encoder.blocks) < 2
        ):
            raise ValueError(
                "VisionZip requires the native eval non-CLS SigLIP branch and projector"
            )
        self.backend, self.config, self.projector = backend, config, projector
        self.size = encoder.patch_embed.num_patches
        self.block_index = len(encoder.blocks) - 2
        self.scale = encoder.blocks[self.block_index].attn.scale
        self.adapter = self.probe = None
        self.patches = ForwardPatches()
        self.closed = False
        self.reset(None)
        if config.enabled and config.keep_ratio < 1:
            if getattr(projector, "_evb_projector_compression", None) is not None:
                raise ValueError("A projector compressor already owns this instance")
            projector._evb_projector_compression = self
            try:
                self.probe = VisionAttentionProbe(encoder.blocks[self.block_index].attn)
                self.patches.wrap(projector, self._project)
            except Exception:
                self.close()
                raise

    def reset(self, context):
        if self.probe is not None:
            self.probe.clear()
        self.query_open = False
        self.compressed = self.selected = False
        self.anchors = None
        self.input_count = self.output_count = None
        self.initial_visual = self.final_visual = self.nonvisual = None
        self.layer_sizes = []

    def before_query(self, prepared):
        if self.closed:
            raise RuntimeError("VisionZip method is closed")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def _project(self, original):
        def forward(features):
            try:
                if not self.query_open or self.compressed or torch.is_grad_enabled():
                    raise RuntimeError(
                        "VisionZip requires one native inference projector call per query"
                    )
                if features.ndim != 3 or features.shape[0] != 1 or features.shape[1] % self.size:
                    raise ValueError("Native fused projector patch geometry changed")
                views = features.shape[1] // self.size
                evidence = self.probe.take()
                if len(evidence) != views:
                    raise RuntimeError("Expected one native SigLIP Q/K pair per camera")
                target = max(2, int(self.size * self.config.keep_ratio))
                compressed, anchors = [], []
                for view, (query, key) in enumerate(evidence):
                    if query.shape[-2] != self.size:
                        raise ValueError("Native SigLIP patch count changed")
                    importance, metric = visual_evidence(query, key, self.scale)
                    reduced, positions = compress_view(
                        features[0, view * self.size : (view + 1) * self.size],
                        importance,
                        metric,
                        target,
                        self.config.dominant_ratio,
                    )
                    compressed.append(reduced)
                    anchors.append(positions + view * self.size)
                self.anchors = torch.cat(anchors)
                # Native pointwise MLP executes only on compressed PRE-projection rows.
                projected = original(torch.cat(compressed).unsqueeze(0))
                if projected.ndim != 3 or projected.shape[:2] != (1, target * views):
                    raise ValueError("Native projector changed the patch dimension")
                # Shape-only bridge to unchanged native sequence assembly. Zero rows
                # are removed before the first LLM block, never served as extra context.
                result = projected.new_zeros((1, features.shape[1], projected.shape[-1]))
                result[0, self.anchors] = projected[0]
                self.input_count, self.output_count = features.shape[1], target * views
                self.compressed = True
                return result
            except Exception:
                self.reset(None)
                raise

        return forward

    def start(self, state):
        self.initial_visual = state.layout.views * state.layout.patches_per_view
        self.final_visual = self.initial_visual
        self.nonvisual = len(state.positions) - self.initial_visual
        if self.probe is not None and (
            not self.compressed
            or self.input_count != self.initial_visual
            or state.layout.patches_per_view != self.size
        ):
            raise RuntimeError("Native projector and decoder visual layouts disagree")

    def before_layer(self, layer, state):
        keep = None
        if layer == 0 and self.probe is not None:
            if self.selected or self.anchors is None:
                raise RuntimeError("VisionZip selection is missing or repeated")
            keep = ~state.layout.visual_mask(state.positions)
            keep |= torch.isin(state.positions, self.anchors + 1)
            self.final_visual = self.output_count
            self.selected = True
            self.anchors = None
        self.layer_sizes.append(self.nonvisual + self.final_visual)
        return keep

    def finish(self, state):
        if self.probe is not None:
            self.probe.clear()
        self.query_open = False
        self.backend.query_diagnostics.update(
            visual_tokens_final=self.final_visual,
            layer_sequence_lengths=list(self.layer_sizes),
            visionzip_selected=self.selected,
            visionzip_projector_input_tokens=self.output_count,
            visionzip_native_projector_tokens=self.input_count,
        )

    def abort(self):
        self.reset(None)

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self.patches.close()
        if self.probe is not None:
            self.probe.close()
            self.probe = None
        if getattr(self.projector, "_evb_projector_compression", None) is self:
            del self.projector._evb_projector_compression
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="VisionZip training-free core with released contextual aggregation",
            reference="arXiv:2412.04467v2 sections 2.3 and A.2, algorithms 2 and 3",
            reference_commit="8f86b55c6f000eb033e6912538af2dd7dcb30502",
            implementation_basis="independent equations; explicit released aggregation convention",
            config=asdict(self.config),
            requires_training=False,
            evidence="native SigLIP penultimate received attention and mean-head keys",
            feature_block_index=self.block_index,
            contextual="uniform residual targets; target + mean(assigned non-target features)",
            placement="same groups on fused DINO/SigLIP features BEFORE sparse native projector",
            positions="latest-member native anchors, sorted; original masks and all nonvisual rows",
            numerics="FP32 scoring/aggregation outside autocast; stable ties; epsilon 1e-12",
            excluded=["projector fine-tuning", "CLIP CLS insertion", "published VLM scores"],
        )
