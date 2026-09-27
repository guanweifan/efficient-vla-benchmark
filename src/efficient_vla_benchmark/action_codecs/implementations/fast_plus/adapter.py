"""Released FAST+ DCT/BPE with explicit shape and no silent decoder fallback."""

from pathlib import Path

import numpy as np
from transformers import AutoProcessor

from ...core import Packet, actions_checked
from ...loading import verified_assets
from ...normalization import normalize, unnormalize


class FastPlus:
    def __init__(self, *, assets, calibration, sources=None, device="cpu", corpus=None):
        self.assets = verified_assets(
            assets, "physical-intelligence/fast", "ec4d7aa71691cac0b8bed6942be45684db2110f4"
        )
        self.bounds = calibration["quantiles"]
        self.calibration_sha256 = calibration["corpus_sha256"]
        self.processor = AutoProcessor.from_pretrained(
            str(Path(assets)), trust_remote_code=True, local_files_only=True
        )
        if self.processor.vocab_size != 2048 or self.processor.scale != 10:
            raise ValueError("Unexpected released FAST+ quantizer")

    def encode(self, actions, context):
        actions = normalize(
            actions_checked(actions), self.bounds[context.suite], signed_gripper=True
        )
        ids = self.processor(actions)[0]
        return Packet(np.asarray(ids, dtype=np.int64), 2048, len(actions))

    def decode(self, packet, context):
        decoded = self.processor.bpe_tokenizer.decode(packet.ids.tolist())
        if len(decoded) != packet.horizon * 7:
            raise ValueError("FAST+ invalid coefficient count; zero-action fallback prohibited")
        output = self.processor.decode(
            [packet.ids.tolist()], time_horizon=packet.horizon, action_dim=7
        )[0]
        return unnormalize(output, self.bounds[context.suite], signed_gripper=True)

    def synchronize(self):
        pass  # Released scipy DCT and BPE execute on the host.

    def metadata(self):
        return {
            "codec": "fast_plus",
            "assets": self.assets,
            "normalization": "calibration-suite q01/q99; signed open-gripper",
            "calibration_corpus_sha256": self.calibration_sha256,
            "quantiles": self.bounds,
            "scale": 10,
            "min_token": self.processor.min_token,
            "device": "cpu",
            "side_information": "suite and horizon only",
        }

    def close(self):
        pass
