"""Released SpatialActionTokenizer grids, without the VLM or learned token embeddings."""

import copy
import json
from pathlib import Path

import numpy as np

from ...core import Packet, actions_checked
from ...loading import load_module, verified_assets, verified_sources
from ...normalization import normalize, unnormalize


class TokenNames:
    """Bijective action-only IDs; no linguistic tokenization is measured."""

    def __init__(self):
        self.names = {}
        self.vocab_size = 0

    def __len__(self):
        return len(self.names)

    def add_tokens(self, names, special_tokens=True):
        before = len(self.names)
        for name in names:
            if name not in self.names:
                self.names[name] = len(self.names)
        return len(self.names) - before

    def convert_tokens_to_ids(self, name):
        return self.names[name]


class SpatialVla:
    def __init__(self, *, assets, sources, calibration, corpus=None, device="cpu"):
        if device != "cpu":
            raise ValueError("SpatialVLA quantization uses native NumPy CPU operations")
        self.assets = verified_assets(
            assets,
            "IPEC-COMMUNITY/spatialvla-4b-224-pt",
            "886c4557dfde53b44c0f74fde7d89bf83a26c3cd",
        )
        self.sources = verified_sources(
            sources, "SpatialVLA/SpatialVLA", "18fd74b2a633ec8d9ec7aadcd803969555cc9fbd"
        )
        cfg = json.loads((Path(assets) / "processor_config.json").read_text())
        module = load_module(Path(sources) / "model/action_tokenizer.py", "evb_spatial_codec")
        self.names = TokenNames()
        self.tokenizer = module.SpatialActionTokenizer(
            self.names,
            num_bins=cfg["action_config"]["num_bins"],
            bin_policy=cfg["bin_policy"],
            use_spherical=cfg["action_config"]["use_spherical"],
        )
        if self.tokenizer.vocab_size != 8194:
            raise ValueError("Unexpected released SpatialVLA action alphabet")
        self.calibration = copy.deepcopy(calibration)

    def encode(self, actions, context):
        value = normalize(
            actions_checked(actions),
            self.calibration["quantiles"][context.suite],
            signed_gripper=False,
        )
        names = self.tokenizer(value)
        ids = np.array([self.names.names[s] for s in names.flat], dtype=np.int64).reshape(-1, 3)
        return Packet(ids, 8194, len(value))

    def decode(self, packet, context):
        if packet.vocabulary != 8194 or packet.ids.shape != (packet.horizon, 3):
            raise ValueError("SpatialVLA packet mismatch")
        if (
            np.any(packet.ids[:, 0] >= 4096)
            or np.any((packet.ids[:, 1] < 4096) | (packet.ids[:, 1] >= 8192))
            or np.any(packet.ids[:, 2] < 8192)
        ):
            raise ValueError("SpatialVLA token in the wrong positional sub-alphabet")
        value = self.tokenizer.decode_token_ids_to_actions(packet.ids)
        return unnormalize(
            value, self.calibration["quantiles"][context.suite], signed_gripper=False
        )

    def synchronize(self):
        pass

    def metadata(self):
        return {
            "codec": "spatialvla",
            "comparison_scope": "coordinate_transfer",
            "device": "cpu",
            "precision": "released NumPy grids",
            "assets": self.assets,
            "sources": self.sources,
            "calibration": copy.deepcopy(self.calibration),
            "side_information": "horizon, fixed pretrained grids and suite normalization",
            "payload": "three disjoint action IDs in joint vocabulary8194; 14 bits each",
            "scope": "spatial quantization only; no VLM or learned embedding benefit claim",
            "representation": "normalized raw OSC coordinates; not physical Euler reconstruction",
            "pretraining_exposure": "released pretrained Gaussian grids; no LIBERO bin refitting",
        }

    def close(self):
        self.tokenizer = None
