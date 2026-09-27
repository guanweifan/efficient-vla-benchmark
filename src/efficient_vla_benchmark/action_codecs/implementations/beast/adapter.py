"""Native closed-form B-spline fitting; freeze coefficient bounds before evaluation."""

from pathlib import Path

import torch

from ...core import Packet, actions_checked
from ...loading import load_module, verified_assets


class Beast:
    def __init__(
        self,
        *,
        assets,
        calibration,
        corpus,
        sources=None,
        device="cpu",
        num_basis=10,
        degree=4,
        gripper_zero_order=False,
    ):
        if type(num_basis) is not int or not degree < num_basis <= 40 or degree < 0:
            raise ValueError("Expected degree < num_basis <= horizon")
        if type(gripper_zero_order) is not bool:
            raise ValueError("gripper_zero_order must be boolean")
        if corpus.manifest["horizon"] != 40:
            raise ValueError("Current BEAST comparison requires forty-frame windows")
        self.assets = verified_assets(
            assets, "zhouhongyi/beast", "ec3cb260baaa780d52785e3b64fdb2fb7690c1a9"
        )
        self.device = torch.device(device)
        self.calibration_sha256 = calibration["corpus_sha256"]
        self.config = dict(
            num_basis=num_basis, degree=degree, gripper_zero_order=gripper_zero_order
        )
        source = load_module(Path(assets) / "beast.py", "evb_beast_reference")
        self.tokenizer = source.BeastTokenizer(
            num_dof=7,
            num_basis=num_basis,
            seq_len=40,
            degree_p=degree,
            gripper_zero_order=gripper_zero_order,
            device=str(self.device),
        ).eval()
        with torch.inference_mode():
            for row in corpus.rows("calibration"):
                actions, _ = corpus.window(row)
                self.tokenizer.encode_discrete(
                    torch.as_tensor(actions[None], device=self.device), update_bounds=True
                )
        self.frozen_bounds = (self.tokenizer.w_min.clone(), self.tokenizer.w_max.clone())
        self.synchronize()

    def encode(self, actions, context):
        value = actions_checked(actions, 40)
        with torch.inference_mode():
            tokens = self.tokenizer.encode_discrete(
                torch.as_tensor(value[None], device=self.device), update_bounds=False
            )
        return Packet(tokens[0].cpu().numpy(), 256, 40)

    def decode(self, packet, context):
        if packet.horizon != 40 or packet.ids.size != 7 * self.config["num_basis"]:
            raise ValueError("BEAST packet shape mismatch")
        tokens = torch.tensor(packet.ids[None], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.tokenizer.decode_discrete(tokens)[0].float().cpu().numpy()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self):
        if any(
            not torch.equal(a, b)
            for a, b in zip(self.frozen_bounds, (self.tokenizer.w_min, self.tokenizer.w_max))
        ):
            raise ValueError("BEAST bounds changed after calibration")
        return {
            "codec": "beast",
            "assets": self.assets,
            "config": self.config,
            "precision": "float32",
            "device": self.device.type,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "calibration_corpus_sha256": self.calibration_sha256,
            "coefficient_min": self.frozen_bounds[0].cpu().tolist(),
            "coefficient_max": self.frozen_bounds[1].cpu().tolist(),
            "normalization": "raw commands, calibration-only native coefficient bounds",
            "side_information": "horizon only; no enforced initial-position constraint",
        }

    def close(self):
        self.tokenizer = None
