from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe

from ...selection import REFERENCE_REVISION, select_visual, stage_budget, text_raters


class SparseVLM(InferenceMethod, TokenStrategy):
    name = "sparsevlm"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = None
        self.probe_handles = {}
        self.reset(None)
        if config.enabled:
            decoder = backend.model_access.decoder
            if max(config.pruning_layers) >= len(decoder.layers) - 1:
                raise ValueError("SparseVLM pruning must leave a subsequent decoder layer")
            if any(
                config.priority_heads > decoder.layers[i].self_attn.num_heads
                for i in config.pruning_layers
            ):
                raise ValueError("SparseVLM priority_heads exceeds native attention heads")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                for layer in config.pruning_layers:
                    self.probe_handles[layer] = NativeAttentionProbe(
                        decoder.layers[layer].self_attn,
                        lambda: self.raters,
                        lambda: self.adapter.state.positions,
                        per_head=True,
                        gravity_correction=True,
                    )
            except Exception:
                self.close()
                raise

    def reset(self, context):
        self.raters = None
        self.stages = []
        for probe in self.probe_handles.values():
            probe.clear()

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def before_layer(self, layer, state):
        if layer == 0:
            visual = state.layout.visual_mask(state.positions)
            text = (state.positions >= state.layout.visual_end) & (
                state.positions < state.layout.cognition_position
            )
            self.original_visual_count = int(visual.sum())
            self.raters = state.positions[text][
                text_raters(state.hidden[0, visual], state.hidden[0, text])
            ]

    def after_layer(self, layer, state):
        if layer not in self.probe_handles:
            return
        visual = state.layout.visual_mask(state.positions)
        probe = self.probe_handles[layer]
        try:
            ids, stats = select_visual(
                probe.probabilities,
                visual,
                stage_budget(
                    self.original_visual_count, self.config.pruning_layers.index(layer), self.config
                ),
                self.config.priority_heads,
            )
        finally:
            probe.clear()
        keep = ~visual
        keep[visual.nonzero().flatten()[ids]] = True
        self.stages.append({"layer": layer, **stats})
        return keep

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
            sparsevlm_version="2.0",
            sparsevlm_stages=self.stages,
            text_raters=int(self.raters.numel()),
        )
        self.raters = None

    def abort(self):
        self.reset(None)

    def close(self):
        for probe in self.probe_handles.values():
            probe.close()
        self.probe_handles.clear()
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "SparseVLM+ V2.0: gravity correction, priority heads, fixed budgets",
            "reference_revision": REFERENCE_REVISION,
            "recycling": False,
            "positions": "original native positions; no contiguous renumbering",
            "budget_mapping": "author 576-patch budgets scaled to original native visual count",
            "text": "native prompt raters; strict above-mean relevance; no cognition row",
            "requires_training": False,
            "config": asdict(self.config),
            "attention": "native causal SDPA; protected cognition EOS and unchanged DDIM",
        }
