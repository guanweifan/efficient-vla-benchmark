"""Depth regions, persistent groups, gradual BSM and region-level recovery."""

from collections import deque

import torch
from torch.nn import functional as F


def region_partition(depth, protected, count):
    ids = (~protected).nonzero().flatten()
    labels = torch.full_like(depth, -1, dtype=torch.long)
    if not len(ids):
        return labels
    values = depth[ids].float()
    count = min(count, len(ids))
    quantiles = torch.linspace(0, 1, count + 2, device=depth.device)[1:-1]
    centers = torch.quantile(values, quantiles)
    for _ in range(10):
        assignment = (values[:, None] - centers[None]).abs().argmin(-1)
        updated = []
        for index in range(count):
            mask = assignment == index
            updated.append(values[mask].mean() if bool(mask.any()) else centers[index])
        centers = torch.stack(updated)
    labels[ids] = (values[:, None] - centers[None]).abs().argmin(-1)
    return labels


class ProgressiveDepthMerge:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.depth_history = deque(maxlen=self.config.merge_window)
        self.saliency = []
        self.groups = self.regions = self.protected = None
        self.ages, self.targets = {}, {}
        self.depth = self.edge = None
        self.dynamic_regions = 0

    @property
    def warming(self):
        return self.regions is None

    def observe_depth(self, depth, edges):
        self.depth, self.edge = depth, edges
        self.depth_history.append(depth.clone())
        if self.groups is None:
            self.groups = torch.arange(len(depth), device=depth.device)

    def collect_saliency(self, saliency):
        if not self.warming:
            return
        self.saliency.append(saliency.detach().float())
        if len(self.saliency) < self.config.warmup_queries:
            return
        importance = torch.stack(self.saliency).mean(0)
        self.protected = (
            importance > importance.mean() + importance.std(unbiased=False)
        ) | self.edge
        self.regions = region_partition(self.depth, self.protected, self.config.depth_clusters)
        region_ids = self.regions.unique().tolist()
        region_ids = [index for index in region_ids if index >= 0]
        if region_ids:
            means = torch.stack([self.depth[self.regions == index].mean() for index in region_ids])
            span = means.max() - means.min()
            fractions = (means - means.min()) / span.clamp_min(1e-12)
            ratios = (
                self.config.min_merge_ratio
                + (self.config.max_merge_ratio - self.config.min_merge_ratio) * fractions
            )
            for index, ratio in zip(region_ids, ratios.tolist()):
                size = int((self.regions == index).sum())
                self.targets[index] = min(size - 1, int(size * ratio))
                self.ages[index] = -1
        self.saliency.clear()

    def _merge_to(self, features, members, target):
        original_count = len(members)
        while True:
            ids = self.groups[members].unique(sorted=True)
            remaining = target - (original_count - len(ids))
            if remaining <= 0 or len(ids) <= 1:
                return
            membership = self.groups[None, :] == ids[:, None]
            mass = membership.sum(-1)
            values = (membership.float() @ features.float()) / mass[:, None]
            unit = F.normalize(values, dim=-1)
            source, destination = ids[::2], ids[1::2]
            similarity = unit[::2] @ unit[1::2].T
            score, matches = similarity.max(-1)
            take = min(remaining, len(source), len(ids) // 2)
            chosen = torch.argsort(score, descending=True, stable=True)[:take]
            mapping = torch.arange(len(self.groups), device=features.device)
            mapping[source[chosen]] = destination[matches[chosen]]
            self.groups = mapping[self.groups]

    def reduce(self, features):
        if self.groups is None or len(features) != len(self.groups):
            raise ValueError("Depth and projected primary-view patches must align")
        self.dynamic_regions = 0
        if not self.warming:
            window = torch.stack(tuple(self.depth_history))
            dynamic = window.amax(0) - window.amin(0) > self.config.static_threshold_m
            for region, target in self.targets.items():
                members = (self.regions == region).nonzero().flatten()
                if float(dynamic[members].float().mean()) > self.config.dynamic_fraction:
                    self.groups[members] = members
                    self.ages[region] = -1
                    self.dynamic_regions += 1
                    continue
                self.ages[region] += 1
                scheduled = int(
                    min(self.ages[region], self.config.merge_window)
                    / self.config.merge_window
                    * target
                )
                self._merge_to(features, members, scheduled)
        ids = self.groups.unique(sorted=True)
        membership = self.groups[None, :] == ids[:, None]
        mass = membership.sum(-1)
        merged = ((membership.float() @ features.float()) / mass[:, None]).to(features.dtype)
        original = torch.arange(len(features), device=features.device)
        anchors = original[None, :].expand(len(ids), -1).masked_fill(~membership, -1).amax(-1)
        order = anchors.argsort()
        return anchors[order], merged[order]

    def diagnostics(self):
        return {
            "depth_warmup": self.warming,
            "depth_dynamic_regions": self.dynamic_regions,
            "depth_primary_tokens": int(self.groups.unique().numel()),
            "depth_protected_tokens": None if self.protected is None else int(self.protected.sum()),
        }
