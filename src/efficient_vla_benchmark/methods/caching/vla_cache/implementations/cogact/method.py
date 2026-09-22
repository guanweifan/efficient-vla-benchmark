"""Single-view frame and prompt boundaries for the complete shared cache policy."""

from efficient_vla_backend_cogact.reuse import ReuseStrategy

from ...method import VLACacheCore


class VLACache(VLACacheCore, ReuseStrategy):
    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        decoder = backend.model_access.decoder
        if len(decoder.layers) != 32:
            raise ValueError("VLA-Cache CogACT requires the native 32-layer Llama decoder")
        dtype = next(decoder.parameters()).dtype
        super().__init__(backend, config, views=1, dtype=dtype)
        self.adapter = backend.model_access.attach_reuse_strategy(self)

    def before_query(self, prepared):
        image = self.backend.model_access.policy_image(prepared["image"])
        self.prepare_frame(prepared["instruction"], [image])

    def text_bounds(self, layout):
        return layout.visual_end, layout.cognition_position

    def metadata(self):
        return {
            **super().metadata(),
            "scope": "complete static/task-aware entropy-scheduled KV/MLP reuse",
            "camera_views": 1,
            "attention_mask": "each original query's causal visibility to all retained keys",
            "cognition": "always recomputed; native last-row readout and DDIM sampler",
            "frame_reference": "previous model query; native policy RGB geometry",
            "reference_commit": "a4909880573868dee2769343d52e793c0341678b",
        }
