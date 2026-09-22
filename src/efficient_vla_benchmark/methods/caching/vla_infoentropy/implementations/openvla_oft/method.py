"""OFT image/text layout for the shared information-entropy cache policy."""

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...method import InfoEntropyCore
from ...method import attention_information as attention_information
from ...method import cache_schedule as cache_schedule


class VLAInfoEntropy(InfoEntropyCore, DecoderStrategy):
    def __init__(self, backend, config):
        super().__init__(backend, config, views=2, dtype=backend.dtype)
        if self.active:
            self.adapter = DecoderAdapter(backend.model.language_model.model, self)

    def before_query(self, prepared):
        self.prepare_frame(prepared["observation"].instruction, prepared["images"])

    def text_bounds(self, layout):
        return layout.text_start, layout.text_end
