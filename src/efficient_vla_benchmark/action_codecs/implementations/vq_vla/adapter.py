"""Pinned native causal VQ-VAE with four stage-local codes per five commands."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from ...core import Packet, actions_checked
from ...loading import load_module, verified_assets, verified_sources
from ...normalization import normalize, unnormalize


class VqVla:
    def __init__(self, *, assets, calibration, corpus, sources, device="cpu"):
        self.assets = verified_assets(
            assets, "VQ-VLA/vq-vla-weight", "cbf998d66004f32010e9d070539d8f6e7eed510f"
        )
        sources = Path(sources)
        self.sources = verified_sources(
            sources, "xiaoxiao0406/VQ-VLA", "7fb78e63f5c0f7baf6d340166ffea527c6514610"
        )
        module = load_module(sources / "prismatic/action_vqvae/__init__.py", "evb_vq_vla")
        self.model = module.ActionVQVAELossWrapper(
            str(sources / "train_vae/scripts/action_vqvae_config"),
            is_eval=True,
            freeze=True,
            use_action_type_pe=True,
            use_time_pe=True,
        )
        # The pinned training export stores an argparse.Namespace beside tensor weights.
        # Allow only known metadata types/scalars, not arbitrary pickle globals.
        safe_metadata = [
            Namespace,
            np.core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float64DType,
            np.dtypes.Float32DType,
        ]
        with torch.serialization.safe_globals(safe_metadata):
            checkpoint = torch.load(
                Path(assets) / "action_tokenizer_weight/all_data_vq.pth",
                map_location="cpu",
                weights_only=True,
            )
        state = checkpoint.get("model", checkpoint)
        if not (checkpoint["args"].use_action_type_pe and checkpoint["args"].use_time_pe):
            raise ValueError("Released all-data checkpoint PE flags differ from its native recipe")
        if not state or any(not k.startswith("vqvae.") for k in state):
            raise ValueError("Unexpected VQ-VLA model-state keys")
        self.model.vqvae.load_state_dict(
            {k.removeprefix("vqvae."): v for k, v in state.items()}, strict=True
        )
        del checkpoint, state
        self.device = torch.device(device)
        self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.bounds = calibration["quantiles"]
        self.calibration_sha256 = calibration["corpus_sha256"]

    def encode(self, actions, context):
        value = normalize(
            actions_checked(actions), self.bounds[context.suite], signed_gripper=False
        )
        if len(value) % 5:
            raise ValueError("VQ-VLA needs whole native five-frame blocks")
        ids = []
        with torch.inference_mode():
            for block in np.split(value, len(value) // 5):
                tensor = torch.as_tensor(block[None], device=self.device)
                ids.append(self.model.get_code(tensor)[0].cpu().numpy())
        return Packet(np.asarray(ids, dtype=np.int64), 256, len(value))

    def decode(self, packet, context):
        if packet.horizon % 5 or packet.ids.shape != (packet.horizon // 5, 4):
            raise ValueError("VQ-VLA native token/block shape mismatch")
        blocks = []
        with torch.inference_mode():
            for tokens in packet.ids:
                tensor = torch.tensor(tokens[None], device=self.device, dtype=torch.long)
                latent = self.model.draw_code_forward(tensor)
                block = self.model.get_action_from_latent(latent)[0].float().cpu().numpy()
                blocks.append(actions_checked(block, 5))
        return unnormalize(np.concatenate(blocks), self.bounds[context.suite], signed_gripper=False)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self):
        return {
            "codec": "vq_vla",
            "assets": self.assets,
            "sources": self.sources,
            "precision": "float32",
            "device": self.device.type,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "native_horizon": 5,
            "positional_encoding": "native action-type and temporal PE; checked training flags",
            "tokens_per_native_block": 4,
            "normalization": "native BOUNDS_Q99 recipe fitted on calibration; 0/1 open gripper",
            "calibration_corpus_sha256": self.calibration_sha256,
            "quantiles": self.bounds,
            "code_ids": "256 stage-local IDs; fixed stage order; no policy LLM-vocabulary offsets",
            "side_information": "suite, horizon and fixed quantizer order only",
            "pretraining_exposure": "released all-data tokenizer may include LIBERO",
        }

    def close(self):
        self.model = None
