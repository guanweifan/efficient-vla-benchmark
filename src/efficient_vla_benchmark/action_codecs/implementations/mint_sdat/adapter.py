"""Released MINT SDAT multiscale residual codec, independent of its policy."""

from pathlib import Path

import numpy as np
import torch
import yaml

from ...core import Packet, actions_checked
from ...loading import load_module, verified_assets, verified_sources


class MintSdat:
    def __init__(self, *, assets, sources, calibration=None, corpus=None, device="cpu"):
        self.assets = verified_assets(
            assets,
            "huangrm/MINT-tokenizer-libero-130",
            "11cbedb05fee2f73b14325fe127bf128ab2e7805",
        )
        self.sources = verified_sources(
            sources, "RenMing-Huang/MINT", "bed9f0c0e2445a123814e89e4ba8c35a9578ce52"
        )
        cfg = yaml.safe_load((Path(assets) / "config.yaml").read_text())
        if cfg["horizon"] != 16 or cfg["quant_cfg"]["patch_nums"] != [1, 2, 4]:
            raise ValueError("Unexpected released MINT horizon/scales")
        module = load_module(Path(sources) / "SDAT/model/__init__.py", "evb_mint_sdat")
        m, q = cfg["model_cfg"], cfg["quant_cfg"]
        # Instantiate directly: upstream training builder globally disables torch
        # initializers. Do not call that builder or mutate any global torch class.
        self.model = module.VQVAE(
            seq_dim=m["action_dim"],
            codebook_size=m["codebook_size"],
            z_channels=m["codebook_dim"],
            ch=m["ch"],
            ch_mult=m["ch_mult"],
            dropout=m["dropout"],
            patchwise=m["patchwise"],
            beta=q["beta"],
            znorm=q["znorm"],
            quant_conv_ks=q["quant_conv_ks"],
            quant_resi=q["quant_resi"],
            share_quant_resi=q["share_quant_resi"],
            patch_nums=q["patch_nums"],
            test_mode=True,
        )
        state = torch.load(Path(assets) / "ms_vqvae.pth", map_location="cpu", weights_only=True)
        if state["cfg"]["dataset_cfg"]["action_scale"] != 1.0:
            raise ValueError("Unsupported released action scale")
        # The author's policy loader selects vae_wo_ddp, not the EMA copy.
        torch.nn.Module.load_state_dict(self.model, state["trainer"]["vae_wo_ddp"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, actions, context):
        value = actions_checked(actions)
        if len(value) % 16:
            raise ValueError("MINT needs whole native sixteen-frame blocks")
        tokens = []
        for block in np.split(value, len(value) // 16):
            indices = self.model.inp_to_idxBl(torch.from_numpy(block).to(self.device)[None])
            tokens.append(torch.cat(indices, dim=1)[0])
        return Packet(torch.stack(tokens).cpu().numpy(), 512, len(value))

    @torch.inference_mode()
    def decode(self, packet, context):
        if (
            packet.vocabulary != 512
            or packet.horizon % 16
            or packet.ids.shape != (packet.horizon // 16, 7)
        ):
            raise ValueError("MINT multiscale packet mismatch")
        values = []
        for ids in packet.ids:
            ids = torch.tensor(ids.copy(), device=self.device)[None]
            quantizer = self.model.quantizer
            accumulated = torch.zeros(1, 32, 4, device=self.device)
            for i, scale_ids in enumerate(torch.split(ids, [1, 2, 4], dim=1)):
                embeddings = quantizer.embedding(scale_ids).transpose(1, 2).contiguous()
                accumulated, _ = quantizer.get_next_autoregressive_input(
                    i, 3, accumulated, embeddings
                )
            value = self.model.patchwise_proj(
                self.model.decoder(self.model.post_quant_conv(accumulated))
            )
            # Native policy uses categorical gripper decoding, not posthoc denoising.
            gripper = value[..., 6:8].argmax(-1, keepdim=True) * 2 - 1
            values.append(torch.cat([value[..., :6], gripper.to(value.dtype)], -1)[0])
        return torch.cat(values).cpu().numpy()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self):
        return {
            "codec": "mint_sdat",
            "assets": self.assets,
            "sources": self.sources,
            "device": self.device.type,
            "precision": "float32",
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "native_horizon": 16,
            "tokens_per_native_block": 7,
            "scales": [1, 2, 4],
            "checkpoint_component": "trainer.vae_wo_ddp",
            "normalization": "released checkpoint LIBERO HDF5 actions, action_scale=1.0",
            "side_information": "horizon only",
            "pretraining_exposure": "released tokenizer includes all five LIBERO suites",
            "scope": "SDAT multiscale residual quantization; no autoregressive policy",
            "frequency_note": "DCT is a training loss; no DCT or inverse DCT at inference",
        }

    def close(self):
        self.model = None
