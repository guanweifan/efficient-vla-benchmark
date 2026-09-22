"""OFT binding for the shared packed-decoder method."""

from ...method import AWQ as DecoderAWQ
from .target import decoder_target


class AWQ(DecoderAWQ):
    def __init__(self, backend, config, artifact_path):
        super().__init__(backend, config, artifact_path, decoder_target(backend))
