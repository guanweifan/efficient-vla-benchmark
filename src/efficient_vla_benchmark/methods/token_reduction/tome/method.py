"""Independent per-encoder ToMe with spatial unmerging at the native readout."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from .merging import attention_with_mass, bipartite_groups, weighted_merge


class ToMeCore(InferenceMethod):
    name = "tome"

    def __init__(self, backend, config, encoders):
        self.backend, self.config, self.encoders = backend, config, encoders
        self.patches = ForwardPatches()
        self.closed = False
        self.reset(None)
        if config.enabled and config.tokens_per_layer:
            try:
                for name, encoder in encoders.items():
                    if encoder.training or type(encoder.num_prefix_tokens) is not int:
                        raise ValueError(
                            "ToMe requires native eval ViTs with declared prefix count"
                        )
                    if getattr(encoder, "_evb_tome", None) is not None:
                        raise ValueError("An encoder already has ToMe attached")
                    encoder._evb_tome = self
                    self.patches.wrap(encoder, self._encoder(name, encoder))
                    for index, block in enumerate(encoder.blocks):
                        if not all(
                            hasattr(block, key)
                            for key in ("ls1", "ls2", "drop_path1", "drop_path2")
                        ):
                            raise ValueError("Unsupported native timm residual block")
                        self.patches.wrap(block, self._block(name, encoder, block, index))
            except Exception:
                self.close()
                raise

    def _encoder(self, name, encoder):
        def wrap(original):
            def forward(images):
                if name in self.state:
                    raise RuntimeError("Nested ToMe encoder forward")
                if torch.is_grad_enabled() or encoder.training:
                    raise RuntimeError("ToMe port supports inference only")
                self.state[name] = {"mass": None, "membership": None, "lengths": []}
                try:
                    sparse = original(images)
                    state = self.state[name]
                    prefix = encoder.num_prefix_tokens
                    if not isinstance(sparse, torch.Tensor) or sparse.ndim != 3:
                        raise ValueError("Expected native penultimate patch-only readout")
                    if sparse.shape[1] != state["mass"].shape[1] - prefix:
                        raise ValueError("Readout and retained-token membership disagree")
                    mapping = state["membership"][:, prefix:] - prefix
                    dense = sparse.gather(1, mapping[..., None].expand(-1, -1, sparse.shape[-1]))
                    self.vision_lengths.setdefault(name, []).append(list(state["lengths"]))
                    self.retained.setdefault(name, []).append(sparse.shape[1])
                    return dense
                finally:
                    del self.state[name]

            return forward

        return wrap

    def _block(self, name, encoder, block, index):
        def wrap(original):
            def forward(hidden):
                state = self.state.get(name)
                if state is None:
                    raise RuntimeError("ToMe blocks require their owning encoder call")
                batch, length, _ = hidden.shape
                prefix = encoder.num_prefix_tokens
                state["lengths"].append(length - prefix)
                if state["mass"] is None:
                    state["mass"] = torch.ones((batch, length), device=hidden.device)
                    state["membership"] = torch.arange(length, device=hidden.device).expand(
                        batch, -1
                    )
                # Native get_intermediate_layers executes the final block but serves
                # the penultimate output. Keep this unused work without changing the
                # served readout's membership after it was captured.
                if index == len(encoder.blocks) - 1:
                    return original(hidden)
                attention, keys = attention_with_mass(
                    block.attn, block.norm1(hidden), state["mass"]
                )
                updated = hidden + block.drop_path1(block.ls1(attention))
                groups, count = bipartite_groups(keys, self.config.tokens_per_layer, prefix)
                updated, state["mass"] = weighted_merge(updated, state["mass"], groups, count)
                state["membership"] = groups.gather(1, state["membership"])
                return updated + block.drop_path2(block.ls2(block.mlp(block.norm2(updated))))

            return forward

        return wrap

    def reset(self, context):
        self.state = {}
        self.vision_lengths = {}
        self.retained = {}

    def before_query(self, prepared):
        if self.closed:
            raise RuntimeError("ToMe instance is closed")
        self.reset(None)
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.backend.query_diagnostics.update(
            tome_vision_layer_patch_counts=self.vision_lengths,
            tome_vision_retained=self.retained,
            tome_readout="unmerge each encoder to original patch membership before fusion",
        )

    def close(self):
        self.patches.close()
        for encoder in self.encoders.values():
            if getattr(encoder, "_evb_tome", None) is self:
                del encoder._evb_tome
        self.reset(None)
        self.closed = True

    def metadata(self):
        return {
            "scope": "ToMe vision-block matching, mass-weighted merging and proportional attention",
            "reference": "facebookresearch/ToMe@af95e4b1befa172dadccd8c81e223b10090f9579",
            "implementation_basis": "independent paper equations; no vendored runtime",
            "config": asdict(self.config),
            "requires_training": False,
            "readout": "per-encoder unmerge to original patch grid before native paired fusion",
            "boundaries": "native projector, full LLM context, action head and cadence unchanged",
            "schedule": "constant removal per served vision block, capped by bipartite size",
            "ordering": "released unmerged-A then B convention; stable score ties",
            "transfer": "general vision baseline, not a VLA-specific published method",
        }
