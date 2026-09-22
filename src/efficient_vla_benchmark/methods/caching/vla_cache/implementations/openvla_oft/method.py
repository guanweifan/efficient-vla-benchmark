"""OFT camera and decoder boundaries for the shared complete VLA-Cache core."""

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...method import VLACacheCore
from ...selection import reuse_schedule as reuse_schedule
from ...selection import static_patch_ids as static_patch_ids


class VLACache(VLACacheCore, DecoderStrategy):
    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        decoder = backend.model.language_model.model
        if len(decoder.layers) != 32:
            raise ValueError("VLA-Cache presets require the 32-layer OFT backbone")
        super().__init__(backend, config, views=2, dtype=backend.dtype)
        self.adapter = DecoderAdapter(decoder, self)

    def before_query(self, prepared):
        self.prepare_frame(prepared["observation"].instruction, prepared["images"])

    def text_bounds(self, layout):
        return layout.text_start, layout.text_end
