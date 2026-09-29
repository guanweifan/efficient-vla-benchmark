"""One ILSA core, with native decoder access supplied by explicit bindings."""

from dataclasses import asdict

from efficient_vla_benchmark.core.method import InferenceMethod

from .selection import select_tokens, semantic_scores, text_anchors


class ILSAStrategy(InferenceMethod):
    name = "eta_vla_ilsa"

    def __init__(self, backend, config, decoder):
        if config.pruning_layer >= len(decoder.layers) - 1:
            raise ValueError("ILSA pruning must leave a subsequent decoder layer")
        self.backend, self.config, self.decoder = backend, config, decoder
        self.adapter = None
        self.active = config.enabled and config.keep_ratio < 1
        self.reset(None)

    def reset(self, context):
        self.pending = None
        self.stage = {}
        self.layer_lengths = []

    def abort(self):
        self.pending = None

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def before_layer(self, layer, state):
        self.layer_lengths.append(len(state.positions))
        if layer != self.config.pruning_layer:
            return None
        visual = state.layout.visual_mask(state.positions)
        visual_ids = visual.nonzero().flatten()
        text = self.language_mask(state)
        layer_module = self.decoder.layers[layer]
        # H is the native pre-attention normalized input, i.e. the Q/K input.
        hidden = layer_module.input_layernorm(state.hidden)[0]
        anchors = text_anchors(hidden[visual], hidden[text])
        self.stage = dict(layer=layer, text_anchors=len(anchors))
        if not len(anchors):
            # Strict Eq. 4 can yield no anchors (e.g. one text row). Eq. 6 is then
            # undefined; retain native rows instead of inventing another scorer.
            self.stage.update(empty_anchor_dense_fallback=True, output_tokens=len(visual_ids))
            return None
        text_ids = text.nonzero().flatten()[anchors]
        attention = layer_module.self_attn
        head_dim = attention.head_dim
        heads = attention.num_heads
        kv_heads = attention.num_key_value_heads
        queries = attention.q_proj(hidden[text_ids]).reshape(-1, heads, head_dim).transpose(0, 1)
        keys = attention.k_proj(hidden).reshape(-1, kv_heads, head_dim).transpose(0, 1)
        keys = keys.repeat_interleave(heads // kv_heads, dim=0)
        scores = semantic_scores(queries, keys, visual_ids)
        views = (state.positions[visual] - 1) // state.layout.patches_per_view
        selected, stats = select_tokens(
            hidden[visual], scores, views, self.config.keep_ratio, self.config.recycle_ratio
        )
        self.stage.update(stats)
        keep = ~visual
        keep[visual_ids[selected]] = True
        self.pending = keep
        return None

    def selection_after_layer(self, layer):
        if layer != self.config.pruning_layer:
            return None
        result, self.pending = self.pending, None
        return result

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=list(self.layer_lengths),
            ilsa=dict(self.stage),
        )
        self.abort()

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return dict(
            scope="ETA-VLA ILSA equations 3-8 and Algorithm 1; no temporal fusion or training",
            reference="arXiv:2603.25766v1, section III-C and Table III",
            implementation_basis="independent paper equations; no released-code parity claim",
            config=asdict(self.config),
            requires_training=False,
            scoring="native normalized Q/K input; no RoPE or mask ONLY for semantic probe",
            selection="strict above-mean text anchors; global saliency plus within-view diversity",
            diversity="Eq7 all current view members; global recycling from semantic rejects",
            placement="select after complete native block; no current-block MLP savings claimed",
            conventions=[
                "layer4 is zero-based3",
                "mean heads",
                "FP32 scores and stable index ties",
                "epsilon1e-12 cosine; singleton-view diversity0",
                "empty anchors retain dense rows",
            ],
            exclusions=[
                "TFM",
                "driving trajectory optimizer",
                "training",
                "published NAVSIM scores",
            ],
        )
