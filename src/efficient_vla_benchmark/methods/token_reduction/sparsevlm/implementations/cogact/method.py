from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe
from efficient_vla_benchmark.model_ops.token_reduction import TokenReduction

from ...recycling import prune_and_recycle, text_raters


class SparseVLM(InferenceMethod, TokenStrategy):
    name = "sparsevlm"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = None
        self.probes = {}
        self.reset(None)
        if config.enabled:
            decoder = backend.model_access.decoder
            if max(config.pruning_layers) >= len(decoder.layers) - 1:
                raise ValueError("SparseVLM pruning must leave a subsequent decoder layer")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                for layer in config.pruning_layers:
                    self.probes[layer] = NativeAttentionProbe(
                        decoder.layers[layer].self_attn,
                        lambda: self.raters,
                        lambda: self.adapter.state.positions,
                    )
            except Exception:
                self.close()
                raise

    def reset(self, context):
        self.stages = []
        self.abort()

    def abort(self):
        self.raters = None
        for probe in self.probes.values():
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
            self.raters = state.positions[text][
                text_raters(state.hidden[0, visual], state.hidden[0, text])
            ]

    def after_layer(self, layer, state):
        if layer not in self.probes:
            return
        visual = state.layout.visual_mask(state.positions)
        visual_ids = visual.nonzero().flatten()
        probe = self.probes[layer]
        try:
            ids, features, stats = prune_and_recycle(
                state.hidden[0, visual],
                probe.probabilities[:, visual],
                state.positions[visual],
                self.config,
            )
        finally:
            probe.clear()
        self.stages.append({"layer": layer, **stats})
        keep = ~visual
        keep[visual_ids[ids]] = True
        return TokenReduction(keep, features)

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.layout.visual_count(len(state.positions)),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
            sparsevlm_stages=self.stages,
            text_raters=int(self.raters.numel()),
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
            "scope": "SparseVLM v1 raters, rank-adaptive pruning and density recycling",
            "reference": "arXiv:2410.04417v1; not SparseVLM+ or fixed-budget v1.5",
            "reconstruction": "sum at latest-member anchor; no future-position leakage",
            "preset": "fixed benchmark settings; no LIBERO success fitting",
            "attention": "native causal SDPA; raters exclude cognition EOS",
            "requires_training": False,
            "config": asdict(self.config),
        }
