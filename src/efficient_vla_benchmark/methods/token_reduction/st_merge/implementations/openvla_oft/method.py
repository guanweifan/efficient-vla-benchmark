"""Spatial VLA path; no invented video context or vision RoPE correction."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from efficient_vla_backend_openvla_oft.vision import VisionAttentionProbe
from efficient_vla_backend_openvla_oft.vision_adapter import PairedVisionAdapter, VisionReduction
from torch.nn import functional as F

from efficient_vla_benchmark.core.method import InferenceMethod


def spatial_groups(keys, target, queues, sigma, side=16):
    """Each anchor proposes one neighbor per queue; resolve source collisions.

    Queue partition, collision policy and surviving representative are explicit
    port conventions because the paper does not release these executable details.
    """
    count = keys.shape[0]
    ids = torch.arange(count, device=keys.device)
    anchors = ids[::queues]
    normalized = F.normalize(keys.float(), dim=-1)
    positions = torch.stack((ids // side, ids % side), -1).float()
    sources, destinations, scores = [], [], []
    for queue in range(1, queues):
        candidates = ids[queue::queues]
        distance = (positions[anchors, None] - positions[candidates][None]).square().sum(-1)
        similarity = normalized[anchors] @ normalized[candidates].T
        weighted = similarity * torch.exp(-distance / (2 * sigma**2))
        score, best = weighted.max(-1)
        sources.append(candidates[best])
        destinations.append(anchors)
        scores.append(score)
    source, dest, score = torch.cat(sources), torch.cat(destinations), torch.cat(scores)
    order = torch.argsort(score, descending=True, stable=True)
    source, dest = source[order], dest[order]
    ranks = torch.arange(source.numel(), device=keys.device)
    first = torch.full((count,), source.numel(), device=keys.device, dtype=torch.long)
    first.scatter_reduce_(0, source, ranks, reduce="amin", include_self=True)
    unique = ranks == first[source]
    source, dest = source[unique][: count - target], dest[unique][: count - target]
    representative = ids.clone()
    representative[source] = dest
    kept = ids[representative == ids]
    groups = torch.searchsorted(kept, representative)
    return VisionReduction(groups, kept)


class STMerge(InferenceMethod, DecoderStrategy):
    name = "st_merge"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.active = config.enabled and config.tokens_per_view < 256
        self.vision = self.adapter = self.probe = None
        self.reset(None)
        if self.active:
            backbone = backend.model.vision_backbone
            self.probe = VisionAttentionProbe(
                backbone.fused_featurizer.blocks[config.merging_layer].attn
            )
            self.vision = PairedVisionAdapter(backbone, self.after_vision_block)
            self.adapter = DecoderAdapter(backend.model.language_model.model, self)

    def reset(self, context):
        self.plans = {}
        if self.probe:
            self.probe.clear()

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def after_vision_block(self, view, branch, layer, hidden, prefix, positions, sizes):
        if layer != self.config.merging_layer:
            return None
        if branch == "siglip":
            pairs = self.probe.take()
            if len(pairs) != 1:
                raise RuntimeError("ST-Merge expected exactly one current-view SigLIP key probe")
            keys = pairs[0][1][0, :, prefix:].transpose(0, 1).flatten(1)
            self.plans[view] = spatial_groups(
                keys, self.config.tokens_per_view, self.config.queues, self.config.spatial_sigma
            )
        return self.plans[view]

    def start(self, state):
        self.layer_sizes = []
        if self.vision.kept_positions is None:
            raise RuntimeError("ST-Merge vision pass has not supplied its sparse patch layout")

    def before_layer(self, layer, state):
        if layer == 0:
            return ~state.layout.visual_mask(state.positions) | torch.isin(
                state.positions, self.vision.kept_positions
            )

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {}

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=self.vision.kept_positions.numel(),
            layer_sequence_lengths=self.layer_sizes,
            vision_patch_counts={
                f"view{view}_{branch}": counts
                for (view, branch), counts in self.vision.sequence_lengths.items()
            },
        )

    def close(self):
        for adapter in (self.adapter, self.vision, self.probe):
            if adapter:
                adapter.close()
        self.plans.clear()

    def metadata(self):
        return {
            "scope": "spatial VLA encoder merging; no video temporal merging or vision RoPE",
            "implementation_basis": "paper-based: arXiv:2606.29350v1, not verified author code",
            "config": asdict(self.config),
            "grouping": "row-major interleaved queues; highest-score source collision resolution",
            "fusion": "SigLIP-derived grouping shared with DINOv2; size-weighted features",
            "position_policy": "anchor slots, native dense projector bridge then LLM selection",
            "requires_training": False,
        }
