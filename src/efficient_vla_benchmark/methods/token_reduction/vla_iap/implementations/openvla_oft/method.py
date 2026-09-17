"""Semantic/motion IoU gating with independent Sobel boundary protection."""

from collections import deque
from dataclasses import asdict

import numpy as np
import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy
from torch.nn import functional as F

from efficient_vla_benchmark.core.method import InferenceMethod


def minmax(values, epsilon):
    flat = values.flatten(1)
    normalized = (flat - flat.amin(1, keepdim=True)) / (
        flat.amax(1, keepdim=True) - flat.amin(1, keepdim=True)
    ).clamp_min(epsilon)
    return normalized.reshape_as(values)


def sobel_prior(rgb, epsilon):
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 224, 224):
        raise ValueError("OFT geometric priors require cropped 224x224 RGB views")
    grey = (rgb.float() * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
        1, keepdim=True
    )
    kernel = grey.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradients = F.conv2d(
        F.pad(grey, (1, 1, 1, 1), mode="reflect"), torch.stack([kernel, kernel.T])[:, None]
    )
    magnitude = gradients.square().sum(1, keepdim=True).sqrt()
    patches = F.avg_pool2d(magnitude, 14, stride=14)[:, 0]
    return minmax(patches, epsilon)


def semantic_prior(visual, language, config):
    # Both modalities use the SAME per-view spatial mean in the shared LLM space.
    visual = visual.float()
    center = visual.mean(1, keepdim=True)
    vision = F.normalize(visual - center, dim=-1, eps=config.epsilon)
    text = F.normalize(language.float().mean(0)[None, None] - center, dim=-1, eps=config.epsilon)
    probabilities = ((vision * text).sum(-1) / config.temperature).softmax(-1)
    pooled = F.avg_pool2d(
        probabilities.reshape(-1, 1, 16, 16),
        config.semantic_pool_kernel,
        stride=1,
        padding=config.semantic_pool_kernel // 2,
        count_include_pad=False,
    )[:, 0]
    return minmax(pooled, config.epsilon)


def motion_prior(current, previous, previous_previous, history, config):
    difference = (
        (current.float() - 2 * previous.float() + previous_previous.float())
        .norm(dim=-1)
        .reshape(-1, 16, 16)
    )
    accumulated = (1 - config.history_decay) * difference + config.history_decay * history
    size = config.closing_kernel
    dilated = F.max_pool2d(accumulated[:, None], size, stride=1, padding=size // 2)
    closed = -F.max_pool2d(-dilated, size, stride=1, padding=size // 2)
    radius = config.gaussian_kernel // 2
    coords = torch.arange(-radius, radius + 1, device=current.device, dtype=torch.float32)
    weights = torch.exp(-coords.square() / (2 * config.gaussian_sigma**2))
    weights /= weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None]
    smoothed = F.conv2d(F.pad(closed, (radius,) * 4, mode="replicate"), kernel)[:, 0]
    return minmax(smoothed, config.epsilon), accumulated


def interaction_selection(semantic, motion, edge, config):
    sem, temp, geo = (v.flatten(1) for v in (semantic, motion, edge))
    sem_mean, temp_mean = sem.mean(1, keepdim=True), temp.mean(1, keepdim=True)
    sem_std, temp_std = (
        sem.std(1, keepdim=True, correction=0),
        temp.std(1, keepdim=True, correction=0),
    )
    semantic_mask = sem > sem_mean + config.mask_std_scale * sem_std
    motion_mask = temp > temp_mean + config.mask_std_scale * temp_std
    iou = (semantic_mask & motion_mask).sum(1).float() / (semantic_mask | motion_mask).sum(
        1
    ).clamp_min(1)
    aggressive = iou > config.iou_threshold
    background = (sem < sem_mean + config.background_std_scale * sem_std) & (
        temp < temp_mean + config.background_std_scale * temp_std
    )
    center = sem.argmax(1)
    ids = torch.arange(256, device=sem.device)
    distance_squared = (ids[None] // 16 - center[:, None] // 16).square() + (
        ids[None] % 16 - center[:, None] % 16
    ).square()
    core = semantic_mask & (distance_squared <= config.semantic_radius**2)
    selected = torch.where(aggressive[:, None], core | motion_mask, ~background)
    score = sem + temp + config.edge_weight * geo
    selected |= score > config.geometry_threshold
    # Numerical safety only: a degenerate view retains its best-priority patch.
    empty = ~selected.any(1)
    selected[torch.arange(sem.shape[0], device=sem.device), score.argmax(1)] |= empty
    return selected.flatten(), iou, aggressive


class VLAIAP(InferenceMethod, DecoderStrategy):
    name = "vla_iap"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.requires_control_history = config.enabled
        self.reset(None)
        self.adapter = (
            DecoderAdapter(backend.model.language_model.model, self) if config.enabled else None
        )

    def reset(self, context):
        self.history = deque(maxlen=2)
        self.motion_history = None
        self.layout = None
        self.queries = 0
        self.edge = None

    def before_query(self, prepared):
        self.queries += 1
        self.backend.query_diagnostics = {"query_count": self.queries}
        if self.config.enabled and len(self.history) == 2:
            images = np.stack([np.asarray(image) for image in prepared["images"]])
            rgb = (
                torch.from_numpy(images)
                .to(device=self.backend.device, dtype=torch.float32)
                .permute(0, 3, 1, 2)
                / 255
            )
            self.edge = sobel_prior(rgb, self.config.epsilon)

    def start(self, state):
        if self.layout != state.layout:
            self.history.clear()
            self.motion_history = None
        self.layout = state.layout
        self.warmup = len(self.history) < 2
        self.layer_sizes = []

    def before_layer(self, layer, state):
        if layer != 0:
            self.layer_sizes.append(state.positions.numel())
            return None
        layout, hidden, cfg = state.layout, state.hidden[0], self.config
        visual = hidden[1 : layout.visual_end].reshape(2, 256, -1)
        keep = torch.ones_like(state.positions, dtype=torch.bool)
        if not self.warmup:
            sem = semantic_prior(visual, hidden[layout.text_start : layout.text_end], cfg)
            motion, self.motion_history = motion_prior(
                visual, self.history[-1], self.history[-2], self.motion_history, cfg
            )
            selected, iou, aggressive = interaction_selection(sem, motion, self.edge, cfg)
            keep[1 : layout.visual_end] = selected
            self.backend.query_diagnostics.update(
                iou_per_view=iou.tolist(), aggressive_per_view=aggressive.tolist()
            )
        else:
            self.motion_history = visual.new_zeros((2, 16, 16), dtype=torch.float32)
        self.history.append(visual.clone())
        self.edge = None
        self.layer_sizes.append(int(keep.sum().item()))
        return keep

    def finish(self, state):
        self.backend.query_diagnostics.update(
            warmup=self.warmup,
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum().item()),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
        self.reset(None)

    def metadata(self):
        return {
            "scope": "semantic-motion alignment with Sobel geometric token protection",
            "implementation_basis": "paper-based: arXiv:2603.22991v1; not released source",
            "config": asdict(self.config),
            "requires_training": False,
            "temporal_sampling": "successive native policy queries, not extra vision passes",
            "reference_notes": "Untuned constants and text centering policy: see README",
        }
