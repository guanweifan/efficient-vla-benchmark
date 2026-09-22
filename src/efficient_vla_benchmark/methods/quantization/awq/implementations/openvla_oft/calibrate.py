"""Native OFT capture with shared layerwise AWQ calibration."""

from ...calibrate import calibrate as calibrate_decoder
from .target import decoder_target


def calibrate(backend, config, dataset_root, output):
    return calibrate_decoder(backend, config, dataset_root, output, decoder_target(backend))
