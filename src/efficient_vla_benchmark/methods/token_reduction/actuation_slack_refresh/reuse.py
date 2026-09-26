"""ASR pure-attention reuse instrument with an immutable dense KV base."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.decoder import IndexedKVStore

from .state import CleanGate


class CleanKVReuse(InferenceMethod):
    name = "actuation_slack_refresh_kv_reuse"
    requires_control_history = True
    attention_backend = "sdpa"

    def __init__(self, backend, config, layers):
        self.backend, self.config = backend, config
        self.num_layers = layers
        if max(config.gate_layers) >= layers:
            raise ValueError("Gate layer outside native decoder")
        self.active = config.enabled and config.keep_ratio < 1
        self.adapter = None
        self.store = IndexedKVStore()
        self.state = CleanGate(config.keep_ratio)
        self.reset(None)

    def reset(self, context):
        self.state.reset()
        self.store.clear()
        self.last_layout = None
        self.full_positions = None
        self.layer_sizes = []

    def abort(self):
        self.reset(None)

    def start(self, state):
        if self.last_layout != state.layout:
            if self.state.refreshing:
                raise ValueError("Same-frame refresh changed sequence geometry")
            self.reset(None)
        self.last_layout = state.layout
        self.full_positions = torch.arange(state.layout.length, device=state.positions.device)
        self.layer_sizes = []

    def before_query(self, prepared):
        if self.state.refreshing:
            raise RuntimeError("Join dense refresh before serving another query")
        self.backend.query_diagnostics = {"clean_gate_refreshes": self.state.refreshes}

    def wants_auxiliary_query(self):
        return self.active

    def begin_auxiliary_query(self):
        self.state.begin_refresh()
        self.store.clear()

    def end_auxiliary_query(self):
        self.state.end_refresh()

    def auxiliary_diagnostics(self, metrics):
        return {
            "refresh_within_nominal_window": metrics["auxiliary_forward_ms"]
            <= self.config.actuation_window_ms,
            "refresh_nominal_window_ms": self.config.actuation_window_ms,
        }

    def before_layer(self, layer, state):
        if layer == 0:
            return self.state.selection(self.primary_mask(state))

    def key_positions(self, layer, state):
        return self.full_positions

    def update_kv(self, layer, keys, values, state):
        if self.state.refreshing:
            return self.store.update(layer, keys, values, state.positions, state.layout.length)
        if self.state.gate is None:
            return keys, values
        if layer not in self.store.keys:
            raise RuntimeError("Sparse ASR serve requires a complete dense KV base")
        # Out-of-place updates: the served sparse pass NEVER becomes the base of
        # the next query. Its fresh active rows coexist with old dense skipped KV.
        return (
            self.store.keys[layer].index_copy(2, state.positions, keys),
            self.store.values[layer].index_copy(2, state.positions, values),
        )

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        if self.state.refreshing and layer in self.config.gate_layers:
            return {"gate": self.gate_queries(state)}
        return {}

    def after_layer(self, layer, state, attention):
        if self.state.refreshing and layer in self.config.gate_layers:
            self.state.record(attention["gate"][:, self.primary_mask(state)].mean(0))

    def finish(self, state):
        if self.state.refreshing:
            if len(self.store.keys) != self.num_layers:
                raise RuntimeError("Incomplete dense KV refresh cannot be published")
            self.state.publish()
            self.backend.query_diagnostics.update(
                auxiliary_visual_tokens_final=self.visual_count(state.layout),
                clean_gate_refreshes_completed=self.state.refreshes,
                auxiliary_gate_source="same-frame dense forward only",
                kv_base_source="same-frame complete dense forward only",
            )
        else:
            skipped = state.layout.length - state.positions.numel()
            self.backend.query_diagnostics.update(
                visual_query_tokens_final=self.visual_count(state.layout) - skipped,
                visual_kv_tokens=self.visual_count(state.layout),
                layer_sequence_lengths=self.layer_sizes.copy(),
                dense_first_chunk=self.state.gate is None,
            )

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "ASR independent pure-attention KV-reuse instrument",
            "exclusions": "VLA-Cache/VLA-Pruner composition and live-signal gate families",
            "reference": "arXiv:2608.00391v1 Sections 3-4, equations 1-2",
            "gate": self.gate_description,
            "preset": "explicit layer-15 convention; not author-code parity",
            "kv_base": "immutable previous dense refresh; sparse updates are not carried forward",
            "schedule": "first serve dense; unconditional same-frame complete auxiliary pass",
            "timing": "all auxiliary, gate, KV storage and splice costs included",
            "requires_training": False,
            "config": asdict(self.config),
        }
