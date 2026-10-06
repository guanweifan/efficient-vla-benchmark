"""Released conditional RVQ with a disclosed, lossless LIBERO command embedding."""

import copy
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from ...core import Packet, actions_checked
from ...loading import load_module, verified_assets, verified_sources
from .representation import command_path, observed_reference, raw_commands


class XTokenizerCodec:
    def __init__(self, *, assets, calibration, corpus, sources, device="cpu"):
        self.assets = verified_assets(
            assets, "x-square-robot/X-Tokenizer", "bb5a47a36079d085d55c303fdf73223bd3dedd1d"
        )
        sources = Path(sources)
        self.sources = verified_sources(
            sources, "X-Square-Robot/X-Tokenizer", "9733990b16a07d51fb8b5d305ad8c16abb57aef2"
        )
        module = load_module(sources / "xtokenizer/__init__.py", "evb_x_tokenizer")
        statistics = importlib.import_module(module.__name__ + ".tools.statistics_builder")
        self.tokenizer = module.XTokenizer.from_pretrained(
            str(Path(assets) / "xtokenizer.pth"),
            device=device,
            robot_types_path=str(sources / "xtokenizer/configs/robot_types.yaml"),
        )
        self.device = self.tokenizer.device
        self.calibration_sha256 = calibration["corpus_sha256"]
        if self.tokenizer.get_canonical_robot_types()["Franka"] != 2:
            raise ValueError("Unexpected released embodiment registry")
        if (
            self.tokenizer.get_action_dim(),
            self.tokenizer.get_num_quantizers(),
            self.tokenizer.get_vocab_size(),
        ) != (26, 4, 2048):
            raise ValueError("Unexpected released canonical quantizer")
        keys = list(self.tokenizer.get_dof_layout())
        expected = (
            "follow_left_ee_cartesian_pos",
            "follow_left_ee_rotation_6D",
            "follow_left_gripper",
        )
        if tuple(k.removesuffix("_relative") for k in keys[:3]) != expected:
            raise ValueError("Native canonical left-arm layout changed")
        accumulator = statistics.StatisticsAccumulator(
            self.tokenizer.get_dof_layout(),
            self.tokenizer.get_predict_action_keys(),
            self.tokenizer.get_obs_state_keys(),
            chunk_size=32,
            frame_interval=1,
        )
        rng_state = random.getstate()
        random.seed(814)  # Native TDigest compaction uses the Python RNG, not gradient training.
        try:
            for row in corpus.rows("calibration"):
                actions, context = corpus.window(row)
                path, reference = command_path(actions, context)
                path = np.concatenate([reference[None], path])
                accumulator.add_episode(
                    {
                        expected[0]: path[:, :3],
                        expected[1]: path[:, 3:9],
                        expected[2]: path[:, 9:10],
                    }
                )
        finally:
            random.setstate(rng_state)
        exported = accumulator.export()
        # Native missing-key fallback is (min=0, delta=1). Leave absent DOFs absent;
        # do not serialize their all-NaN digest blocks as if observed during calibration.
        exported = {k: v for k, v in exported.items() if np.isfinite(v["min"]).all()}
        if any(not np.isfinite(value).all() for v in exported.values() for value in v.values()):
            raise ValueError("Nonfinite statistics for an observed DOF")
        self.tokenizer.statistics = {"Franka": exported}
        self.frozen_statistics = json.dumps(
            self.tokenizer.statistics, sort_keys=True, allow_nan=False
        )
        self.tokenizer.get_normalizer("Franka")

    @staticmethod
    def mask(horizon):
        mask = np.zeros((1, horizon, 26), dtype=bool)
        mask[:, :, :10] = True
        return mask

    def encode(self, actions, context):
        actions = actions_checked(actions)
        if not 8 <= len(actions) <= 64 or len(actions) % 4:
            raise ValueError("Released X-Tokenizer needs an 8..64 horizon divisible by four")
        path, reference = command_path(actions, context)
        with torch.inference_mode():
            ids = self.tokenizer.encode_from_absolute(
                path[None],
                reference[None, None],
                robot_type="Franka",
                dof_mask=self.mask(len(actions)),
                token_order="time_major",
            )
        return Packet(ids[0].cpu().numpy(), 2048, len(actions))

    def decode(self, packet, context):
        if packet.ids.shape != (packet.horizon // 4, 4):
            raise ValueError("X-Tokenizer token shape mismatch")
        reference = observed_reference(context)
        with torch.inference_mode():
            path = (
                self.tokenizer.decode_to_absolute(
                    packet.ids[None].copy(),
                    packet.horizon,
                    reference[None, None],
                    robot_type="Franka",
                    dof_mask=self.mask(packet.horizon),
                    token_order="time_major",
                )[0]
                .cpu()
                .numpy()
            )
        return raw_commands(path, context)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self):
        if (
            json.dumps(self.tokenizer.statistics, sort_keys=True, allow_nan=False)
            != self.frozen_statistics
        ):
            raise ValueError("X-Tokenizer statistics changed after calibration")
        return {
            "codec": "x_tokenizer",
            "comparison_scope": "state_conditioned",
            "assets": self.assets,
            "sources": self.sources,
            "precision": "float32",
            "device": self.device.type,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "calibration_corpus_sha256": self.calibration_sha256,
            "statistics": copy.deepcopy(self.tokenizer.statistics),
            "statistics_chunk_size": 32,
            "statistics_seed": 814,
            "embodiment": "Franka (ID2)",
            "layout": "left arm10 of native26; missing DOFs masked",
            "representation": "command-integrated poses; not future observations or actual motion",
            "conversion": "native OSC0.05m/0.5rad; left SO3 composition; gripper width/0.08m",
            "side_information": (
                "one initial 8-float32 observed state at both ends; excluded from payload bits"
            ),
            "pretraining_exposure": "not certified unseen; adapted command embedding",
        }

    def close(self):
        self.tokenizer = None
