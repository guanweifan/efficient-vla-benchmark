from dataclasses import asdict

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...state import CleanGate


class ActuationSlackRefresh(InferenceMethod, DecoderStrategy):
    name = "actuation_slack_refresh_deletion"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.state = CleanGate(config.keep_ratio)
        self.active = config.enabled and config.keep_ratio < 1
        self.adapter = None
        self.layer_sizes = []
        if self.active:
            decoder = backend.model.language_model.model
            if max(config.gate_layers) >= len(decoder.layers):
                raise ValueError("Gate layer outside native decoder")
            self.adapter = DecoderAdapter(decoder, self)

    def reset(self, context):
        self.state.reset()
        self.layer_sizes = []

    def before_query(self, prepared):
        if self.state.refreshing:
            raise RuntimeError("Dense refresh must finish before another served query")
        self.layer_sizes = []
        self.backend.query_diagnostics = {"clean_gate_refreshes": self.state.refreshes}

    def wants_auxiliary_query(self):
        return self.active

    def begin_auxiliary_query(self):
        self.state.begin_refresh()
        self.layer_sizes = []

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
            return self.state.selection(state.layout.view_mask(state.positions, 0))

    def probes(self, layer, state):
        self.layer_sizes.append(len(state.positions))
        if self.state.refreshing and layer in self.config.gate_layers:
            actions = (state.positions >= state.layout.action_start) & (
                state.positions < state.layout.length - 1
            )
            return {"gate": state.positions[actions]}
        return {}

    def after_layer(self, layer, state, attention):
        if self.state.refreshing and layer in self.config.gate_layers:
            primary = state.layout.view_mask(state.positions, 0)
            self.state.record(attention["gate"][:, primary].mean(0))

    def finish(self, state):
        count = int(state.layout.visual_mask(state.positions).sum())
        if self.state.refreshing:
            self.state.publish()
            self.backend.query_diagnostics.update(
                auxiliary_visual_tokens_final=count,
                clean_gate_refreshes_completed=self.state.refreshes,
                auxiliary_gate_source="same-frame dense forward only",
            )
        else:
            self.backend.query_diagnostics.update(
                visual_tokens_final=count,
                layer_sequence_lengths=self.layer_sizes.copy(),
                dense_first_chunk=self.state.gate is None,
            )

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "ASR independent deletion instrument, unconditional dense clean-gate refresh",
            "exclusions": "KV-reuse instrument and compositions with VLA-Cache/VLA-Pruner",
            "reference": "arXiv:2608.00391v1 Sections 3-4 and Appendix A",
            "gate": "previous dense action-query attention; agentview only; wrist remains full",
            "preset": "explicit layer-15 gate convention; not claimed author-code parity",
            "schedule": "first serve dense; every served chunk schedules one same-frame dense pass",
            "refresh_output": "discarded; never changes the served action chunk",
            "timing": "common synchronized replay includes ALL refresh work; no free-FLOPs claim",
            "latency_diagnostics": "primary_forward_ms excludes preparation; not end-to-end serve",
            "actuation_window": "nominal configured budget; no real-robot deadline claim",
            "requires_training": False,
            "config": asdict(self.config),
        }
