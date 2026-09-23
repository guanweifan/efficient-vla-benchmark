from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe

from ...state import CleanGate


class ActuationSlackRefresh(InferenceMethod, TokenStrategy):
    name = "actuation_slack_refresh_deletion"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.state = CleanGate(config.keep_ratio)
        self.active = config.enabled and config.keep_ratio < 1
        self.adapter = None
        self.probes = {}
        if self.active:
            decoder = backend.model_access.decoder
            if max(config.gate_layers) >= len(decoder.layers):
                raise ValueError("Gate layer outside native decoder")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                for layer in config.gate_layers:
                    self.probes[layer] = NativeAttentionProbe(
                        decoder.layers[layer].self_attn,
                        self._request,
                        lambda: self.adapter.state.positions,
                    )
            except Exception:
                self.close()
                raise

    def _request(self):
        if self.state.refreshing:
            return self.adapter.state.positions[-1:]

    def reset(self, context):
        self.state.reset()
        self.abort()

    def abort(self):
        for probe in self.probes.values():
            probe.clear()

    def before_query(self, prepared):
        if self.state.refreshing:
            raise RuntimeError("Join dense refresh before serving another query")
        self.abort()
        self.backend.query_diagnostics = {"clean_gate_refreshes": self.state.refreshes}

    def wants_auxiliary_query(self):
        return self.active

    def begin_auxiliary_query(self):
        self.state.begin_refresh()

    def end_auxiliary_query(self):
        self.state.end_refresh()
        self.abort()

    def auxiliary_diagnostics(self, metrics):
        return {
            "refresh_within_nominal_window": metrics["auxiliary_forward_ms"]
            <= self.config.actuation_window_ms,
            "refresh_nominal_window_ms": self.config.actuation_window_ms,
        }

    def before_layer(self, layer, state):
        if layer == 0:
            return self.state.selection(state.layout.visual_mask(state.positions))

    def after_layer(self, layer, state):
        if self.state.refreshing and layer in self.probes:
            probe = self.probes[layer]
            try:
                self.state.record(probe.probabilities[0, state.layout.visual_mask(state.positions)])
            finally:
                probe.clear()

    def finish(self, state):
        count = state.layout.visual_count(len(state.positions))
        if self.state.refreshing:
            self.state.publish()
            self.backend.query_diagnostics.update(
                auxiliary_visual_tokens_final=count,
                clean_gate_refreshes_completed=self.state.refreshes,
                auxiliary_gate_source="same-frame dense cognition attention only",
            )
        else:
            self.backend.query_diagnostics.update(
                visual_tokens_final=count,
                layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
                dense_first_chunk=self.state.gate is None,
            )
        self.abort()

    def close(self):
        for probe in self.probes.values():
            probe.close()
        self.probes.clear()
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "unconditional dense clean-gate refresh with independent deletion",
            "exclusions": "KV reuse and compositions with other methods",
            "reference": "arXiv:2608.00391v1 Sections 3-4",
            "gate_adaptation": "native cognition-to-image attention, NOT DiT action-to-patch",
            "core_claim": "gate provenance and unconditional refresh; sampler remains native",
            "schedule": "one same-frame dense pass per served chunk; discarded sampled actions",
            "rng": "auxiliary pass restores native RNG before the next served action sample",
            "timing": "all refresh work included; primary-forward diagnostics exclude preparation",
            "requires_training": False,
            "config": asdict(self.config),
        }
