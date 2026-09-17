"""Training-free layer-zero Q/K selection and observed-motion gating."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...motion import MotionGate


def embedding_scores(layer, text, visual):
    """Released ADP scorer: RMSNorm, pretrained Q/K, raw head-averaged logits.

    No softmax, no RoPE and no hidden full-model probe. All selection stays on GPU.
    """
    attn = layer.self_attn
    q = attn.q_proj(layer.input_layernorm(text))
    k = attn.k_proj(layer.input_layernorm(visual))
    heads, width = attn.num_heads, attn.head_dim
    if attn.num_key_value_heads != heads:
        raise ValueError("The released ADP scorer requires matching Q/K head counts")
    q = q.view(1, -1, heads, width).transpose(1, 2)
    k = k.view(1, -1, heads, width).transpose(1, 2)
    return ((q @ k.transpose(-1, -2)) * width**-0.5).mean((0, 1, 2))


def view_quotas(total, weights, patches):
    # Match the reference's rounding, min-one and per-camera clamp. Very small
    # budgets are raised to the camera count to prevent its nonterminating loop.
    total = max(len(weights), total)
    normalized = [w / sum(weights) for w in weights]
    quotas = [max(1, round(total * w)) for w in normalized]
    order = sorted(range(len(weights)), key=lambda i: normalized[i], reverse=True)
    difference = total - sum(quotas)
    while difference:
        for i in order:
            if difference > 0:
                quotas[i] += 1
                difference -= 1
            elif difference < 0 and quotas[i] > 1:
                quotas[i] -= 1
                difference += 1
            if not difference:
                break
    return [min(q, patches) for q in quotas]


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
