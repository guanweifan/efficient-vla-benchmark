"""OpenVLA's exact 256-boundary digitize/midpoint rule, without LLM token offsets."""

import copy

import numpy as np

from ...core import Packet, actions_checked
from ...normalization import normalize, unnormalize


class UniformBinning:
    def __init__(self, *, calibration, corpus=None, assets=None, sources=None, device="cpu"):
        if device != "cpu":
            raise ValueError("Uniform binning uses its native NumPy CPU implementation")
        self.calibration = copy.deepcopy(calibration)
        self.bins = np.linspace(-1, 1, 256)
        self.centers = (self.bins[:-1] + self.bins[1:]) / 2

    def encode(self, actions, context):
        value = normalize(
            actions_checked(actions),
            self.calibration["quantiles"][context.suite],
            signed_gripper=False,
        )
        ids = np.digitize(np.clip(value, -1, 1), self.bins) - 1
        return Packet(ids, 256, len(value))

    def decode(self, packet, context):
        if packet.vocabulary != 256 or packet.ids.shape != (packet.horizon, 7):
            raise ValueError("Uniform binning packet mismatch")
        value = self.centers[np.clip(packet.ids, 0, 254)]
        return unnormalize(
            value, self.calibration["quantiles"][context.suite], signed_gripper=False
        )

    def synchronize(self):
        pass

    def metadata(self):
        return {
            "codec": "uniform_binning",
            "device": "cpu",
            "precision": "numpy float64 bins",
            "source": "openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9",
            "rule": "256 boundaries, 255 midpoints; last digitized index clips at decode",
            "normalization": "frozen calibration q01/q99; open-gripper [0,1]",
            "calibration": copy.deepcopy(self.calibration),
            "side_information": "horizon and suite normalization",
            "scope": "classical scalar baseline; no LLM or policy measurement",
        }

    def close(self):
        pass
