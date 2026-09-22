"""OFT component binding for portable AWQ artifacts."""

from ...artifact import LINEARS as LINEARS
from ...artifact import load_artifact as load_decoder_artifact
from ...artifact import read_manifest as read_manifest
from ...artifact import replace_linear as replace_linear
from .target import decoder_target


def load_artifact(backend, root, config):
    return load_decoder_artifact(decoder_target(backend), root, config)
