"""Native vision features drive per-view selection before LLM layer zero."""

from dataclasses import asdict

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.vision import VisionFeatureProbe

from .selection import select_current_frame


class CurrentFrameStrategy(InferenceMethod):
    name = "a_mmr_current_frame"

    def __init__(self, backend, config, encoder):
        if (
            encoder.training
            or getattr(encoder, "cls_token", None) is None
            or getattr(encoder, "num_prefix_tokens", 0) < 1
            or len(encoder.blocks) < 2
        ):
            raise ValueError("A-MMR requires the native eval encoder with a genuine CLS prefix")
        self.backend, self.config = backend, config
        self.prefix_tokens = encoder.num_prefix_tokens
        self.block_index = len(encoder.blocks) - 2
        self.adapter = self.probe = None
        self.closed = False
        self.reset(None)
        if config.enabled and config.keep_ratio < 1:
            self.probe = VisionFeatureProbe(encoder.blocks[self.block_index])

    def reset(self, context):
        if self.probe is not None:
            self.probe.clear()
        self.layer_sizes = []
        self.initial_visual = self.final_visual = self.nonvisual = None
        self.selected = False

    def before_query(self, prepared):
        if self.closed:
            raise RuntimeError("A-MMR method is closed")
        self.reset(None)
        self.backend.query_diagnostics = {}

    def start(self, state):
        # Vision executes before LLM start; do not clear the current snapshots here.
        self.initial_visual = state.layout.views * state.layout.patches_per_view
        self.final_visual = self.initial_visual
        self.nonvisual = len(state.positions) - self.initial_visual

    def before_layer(self, layer, state):
        keep = None
        if layer == 0 and self.probe is not None:
            if self.selected:
                raise RuntimeError("A-MMR cannot select twice within one query")
            features = self.probe.take()
            if len(features) != state.layout.views:
                raise RuntimeError("Expected exactly one native DINO snapshot per camera")
            size = state.layout.patches_per_view
            target = max(1, int(size * self.config.keep_ratio))
            keep = ~state.layout.visual_mask(state.positions)
            for view, feature in enumerate(features):
                if feature.shape[:2] != (1, self.prefix_tokens + size):
                    raise ValueError("Native DINO prefix/patch geometry does not match this view")
                begin = 1 + view * size
                rows = (
                    ((state.positions >= begin) & (state.positions < begin + size))
                    .nonzero()
                    .flatten()
                )
                if rows.numel() != size:
                    raise ValueError("A-MMR requires the complete native patch grid")
                selection = select_current_frame(
                    feature[0, self.prefix_tokens :], feature[0, 0], target
                )
                keep[rows] = selection
            self.final_visual = target * state.layout.views
            self.selected = True
        self.layer_sizes.append(self.nonvisual + self.final_visual)
        return keep

    def finish(self, state):
        if self.probe is not None:
            self.probe.clear()
        self.backend.query_diagnostics.update(
            visual_tokens_final=self.final_visual,
            layer_sequence_lengths=list(self.layer_sizes),
            a_mmr_current_frame_selected=self.selected,
        )

    def abort(self):
        self.reset(None)

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        if self.probe is not None:
            self.probe.close()
            self.probe = None
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="A-MMR current-frame spatial-selection core only",
            reference="arXiv:2603.06480v1, sections III-A and III-B, equations 1--2",
            implementation_basis="independent paper equations; no author-code parity claim",
            config=asdict(self.config),
            feature_source="native DINO penultimate block output, CLS row zero and patch rows",
            feature_block_index=self.block_index,
            prefix_tokens=self.prefix_tokens,
            score="min-max CLS cosine saliency times one minus maximum selected-patch cosine",
            budget="floor(patches per view * keep_ratio), minimum one, independently per camera",
            numerics="FP32 cosines; epsilon 1e-6 importance; stable original-position ties",
            placement="native dense vision/projector unchanged; drop paired patch rows before LLM",
            position_policy="original RoPE/masks and all nonvisual action/cognition rows retained",
            requires_training=False,
            excluded=[
                "historical token memory",
                "query-guided history reweighting",
                "navigation scores",
            ],
        )
