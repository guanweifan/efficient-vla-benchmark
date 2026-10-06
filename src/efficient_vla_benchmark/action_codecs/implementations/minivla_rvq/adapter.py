"""Released MiniVLA RVQ codec: MLP encoding, seven residual codebooks, MLP decoding."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...core import Packet, actions_checked
from ...loading import verified_assets
from ...normalization import normalize, unnormalize


class EncoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, value):
        return self.fc(self.encoder(value))


class MiniVlaRvq:
    def __init__(self, *, assets, calibration, corpus=None, sources=None, device="cpu"):
        self.assets = verified_assets(
            assets, "Stanford-ILIAD/pretrain_vq", "30ef2227f97dde1abb6d522ea80af85384235008"
        )
        directory = Path(assets) / "pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512"
        cfg = json.loads((directory / "config.json").read_text())
        if tuple(
            cfg[k]
            for k in (
                "input_dim_h",
                "input_dim_w",
                "n_latent_dims",
                "vqvae_n_embed",
                "vqvae_groups",
                "act_scale",
            )
        ) != (8, 7, 512, 128, 7, 1.0):
            raise ValueError("Unexpected released MiniVLA RVQ architecture")
        state = torch.load(
            directory / "checkpoints/model.pt", map_location="cpu", weights_only=True
        )
        self.device = torch.device(device)
        self.encoder, self.decoder = EncoderMLP(56, 512), EncoderMLP(512, 56)
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.decoder.load_state_dict(state["decoder"], strict=True)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.decoder.to(self.device).eval().requires_grad_(False)
        quantizer = state["vq_embedding"]
        expected = {
            f"layers.{i}._codebook.{suffix}"
            for i in range(7)
            for suffix in ("initted", "cluster_size", "embed_avg", "embed")
        }
        if set(quantizer) != expected:
            raise ValueError("Unexpected learned projection or quantizer state")
        if any(not quantizer[f"layers.{i}._codebook.initted"].all() for i in range(7)):
            raise ValueError("Uninitialized released codebook")
        self.books = torch.stack(
            [quantizer[f"layers.{i}._codebook.embed"][0] for i in range(7)]
        ).to(self.device)
        if self.books.shape != (7, 128, 512):
            raise ValueError("Unexpected residual codebooks")
        self.calibration = copy.deepcopy(calibration)

    @torch.inference_mode()
    def encode(self, actions, context):
        value = normalize(
            actions_checked(actions),
            self.calibration["quantiles"][context.suite],
            signed_gripper=False,
        )
        if len(value) % 8:
            raise ValueError("MiniVLA requires whole eight-frame blocks")
        tokens = []
        for block in np.split(value, len(value) // 8):
            residual = self.encoder(torch.from_numpy(block).to(self.device).reshape(1, 56))
            indices = []
            for book in self.books:
                # Same squared-distance operation order as the released cdist helper.
                distance = (
                    (residual**2).sum(-1, keepdim=True)
                    + (book**2).sum(-1)[None]
                    + torch.einsum("id,jd->ij", residual, book) * -2
                ).sqrt()
                index = (-distance).argmax(-1)
                residual = residual - book[index]
                indices.append(index)
            tokens.append(torch.stack(indices, -1)[0])
        return Packet(torch.stack(tokens).cpu().numpy(), 128, len(value))

    @torch.inference_mode()
    def decode(self, packet, context):
        if (
            packet.vocabulary != 128
            or packet.horizon % 8
            or packet.ids.shape != (packet.horizon // 8, 7)
        ):
            raise ValueError("MiniVLA native packet shape mismatch")
        values = []
        for ids in packet.ids:
            indices = torch.tensor(ids.copy(), device=self.device)
            latent = self.books[torch.arange(7, device=self.device), indices].sum(0, keepdim=True)
            values.append(self.decoder(latent).reshape(8, 7))
        value = torch.cat(values).cpu().numpy()
        return unnormalize(
            value, self.calibration["quantiles"][context.suite], signed_gripper=False
        )

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self):
        return {
            "codec": "minivla_rvq",
            "assets": self.assets,
            "device": self.device.type,
            "precision": "float32",
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "native_horizon": 8,
            "tokens_per_native_block": 7,
            "reference": "jayLEE0301/vq_bet_official@09d4851288ca5deaaa1ab367a208e520f8ee9a84",
            "calibration": copy.deepcopy(self.calibration),
            "pretraining_exposure": "released libero_90 tokenizer; target-suite statistics adapted",
            "scope": "MiniVLA RVQ component; not full VQ-BeT or its policy offsets",
            "side_information": "horizon and suite normalization",
        }

    def close(self):
        self.encoder = self.decoder = self.books = None
