"""Paper equations: spatial expansion, context sampling and soft bipartite merging."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import (
    DecoderAdapter,
    DecoderStrategy,
    TokenReduction,
)
from torch.nn import functional as F

from efficient_vla_benchmark.core.method import InferenceMethod


def expand_seeds(seeds, kernel, threshold, generator):
    views, side, _ = seeds.shape
    x = seeds[:, None].float()
    density = F.conv2d(x, x.new_ones(1, 1, kernel, kernel), padding=kernel // 2)
    dense = F.max_pool2d(
        (density > threshold).float(), kernel, stride=1, padding=kernel // 2
    ).bool()[:, 0]
    result = seeds | dense
    # Parallel sparse expansion: choose from the original (not successively updated) mask.
    if threshold > 1:
        radius = kernel // 2
        offsets = torch.arange(-radius, radius + 1, device=x.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        yy, xx = torch.meshgrid(
            torch.arange(side, device=x.device), torch.arange(side, device=x.device), indexing="ij"
        )
        y = yy.flatten()[:, None] + dy.flatten()
        z = xx.flatten()[:, None] + dx.flatten()
        valid = (y >= 0) & (y < side) & (z >= 0) & (z < side)
        ids = (y.clamp(0, side - 1) * side + z.clamp(0, side - 1)).long()
        available = valid[None] & ~seeds.flatten(1)[:, ids]
        random = torch.rand(available.shape, device=x.device, generator=generator)
        selected = (
            ids[None]
            .expand(views, -1, -1)
            .gather(2, random.masked_fill(~available, -1).argmax(-1, keepdim=True))
            .squeeze(-1)
        )
        active = ((density > 0) & (density < threshold)).flatten(1) & available.any(-1)
        view_ids = torch.arange(views, device=x.device)[:, None].expand_as(active)
        result.flatten(1)[view_ids[active], selected[active]] = True
    return result


def early_selection(visual, language, config, generator):
    views, patches, _ = visual.shape
    side = math_isqrt(patches)
    similarity = F.normalize(visual.float(), dim=-1) @ F.normalize(language.float(), dim=-1).T
    seeds = torch.zeros((views, patches), dtype=torch.bool, device=visual.device)
    seeds.scatter_(1, similarity.argmax(1), True)
    selected = expand_seeds(
        seeds.reshape(views, side, side),
        config.expansion_kernel,
        config.density_threshold,
        generator,
    ).flatten(1)
    count = round(patches * config.context_ratio)
    if count:
        edges = torch.linspace(0, patches, count + 1, device=visual.device).floor().long()
        offsets = torch.rand((views, count), device=visual.device, generator=generator)
        ids = edges[:-1] + (offsets * (edges[1:] - edges[:-1])).long()
        selected.scatter_(1, ids, True)
    return selected.flatten()


def math_isqrt(value):
    import math

    side = math.isqrt(value)
    if side * side != value:
        raise ValueError("TEAM-VLA requires square patch grids")
    return side


def soft_merge(visual, queries, count, epsilon):
    scores = (F.normalize(visual.float(), dim=-1) @ F.normalize(queries.float(), dim=-1).T).amax(1)
    source_ids = scores.topk(count).indices.sort().values
    source_mask = torch.zeros(visual.shape[0], device=visual.device, dtype=torch.bool)
    source_mask[source_ids] = True
    source, target = visual[source_ids].float(), visual[~source_mask].float()
    if target.shape[0] == 0:
        return source_ids, visual[source_ids]
    a = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + epsilon)
    b = target * torch.rsqrt(target.square().mean(-1, keepdim=True) + epsilon)
    weights = ((b @ a.T) / (source.shape[-1] ** 0.5)).softmax(-1)
    merged = (source + weights.T @ target) / (1 + weights.sum(0)[:, None])
    return source_ids, merged.to(visual.dtype)


class TEAMVLA(InferenceMethod, DecoderStrategy):
    name = "team_vla"

    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.active = config.enabled and not (
            config.context_ratio == 1 and config.merge_tokens == 512
        )
        self.reset(None)
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if self.active else None
        )

    def reset(self, context):
        self.generator = torch.Generator(device=self.backend.device)
        self.generator.manual_seed((self.config.seed + (context.seed if context else 0)) % 2**32)
        self.queries = 0

    def before_query(self, prepared):
        self.queries += 1
        self.backend.query_diagnostics = {"query_count": self.queries}

    def start(self, state):
        self.layer_sizes = []
        self.early_count = 512

    def before_layer(self, layer, state):
        if layer == 0:
            layout = state.layout
            hidden = state.hidden[0]
            selected = early_selection(
                hidden[1 : layout.visual_end].reshape(2, 256, -1),
                hidden[layout.text_start : layout.text_end],
                self.config,
                self.generator,
            )
            self.early_count = int(selected.sum().item())
            keep = ~layout.visual_mask(state.positions)
            keep[1 : layout.visual_end] = selected
            self.layer_sizes.append(int(keep.sum().item()))
            return keep
        self.layer_sizes.append(state.positions.numel())

    def after_layer(self, layer, state, attention):
        if layer != self.config.merging_layer:
            return None
        layout, p, hidden = state.layout, state.positions, state.hidden[0]
        visual_mask = layout.visual_mask(p)
        count = min(self.config.merge_tokens, int(visual_mask.sum().item()))
        if count == int(visual_mask.sum().item()):
            return None
        queries = hidden[(p >= layout.text_start) & (p < layout.length - 1)]
        ids, features = soft_merge(hidden[visual_mask], queries, count, self.config.rms_epsilon)
        keep = ~visual_mask
        keep[visual_mask.nonzero().flatten()[ids]] = True
        return TokenReduction(keep, features)

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_early=self.early_count,
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum().item()),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.adapter:
            self.adapter.close()

    def metadata(self):
        return {
            "scope": "early spatial expansion and context sampling plus mid-layer soft merging",
            "implementation_basis": "paper-based: arXiv:2512.09927v1; not released source",
            "config": asdict(self.config),
            "requires_training": False,
            "reference_notes": "See method README for equation and preset ambiguities",
        }
