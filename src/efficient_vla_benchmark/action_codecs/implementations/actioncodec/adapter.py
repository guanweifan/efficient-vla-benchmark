"""Released ActionCodec-Base, tiled native LIBERO blocks without rate conversion."""

import numpy as np
import torch
from transformers import AutoModel

from ...core import Packet, actions_checked
from ...loading import verified_assets


class ReleasedActionCodec:
    def __init__(
        self,
        *,
        assets,
        calibration,
        corpus,
        sources=None,
        device="cpu",
        variant="base",
        quantizers=None,
    ):
        releases = {
            "base": ("ZibinDong/ActionCodec-Base", "65b162110b48303fd409fa08ddb088648993b702"),
            "rvq": ("ZibinDong/ActionCodec-Base-RVQft", "8d6d8a546498d340d62689e5dcd53436adae3af9"),
        }
        if variant not in releases:
            raise ValueError("Unknown released ActionCodec variant")
        self.variant = variant
        self.assets = verified_assets(assets, *releases[variant])
        self.device = torch.device(device)
        self.model, info = AutoModel.from_pretrained(
            str(assets), trust_remote_code=True, local_files_only=True, output_loading_info=True
        )
        if any(
            info.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"Non-strict ActionCodec checkpoint load: {info}")
        self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        keys = sorted(self.model.config.embodiment_config)
        self.embodiment = keys.index("a_franka_libero_20hz")
        cfg = self.model.config.embodiment_config[keys[self.embodiment]]
        if cfg["action_dim"] != 7 or cfg["freq"] * cfg["duration"] != 20:
            raise ValueError("Released LIBERO block differs from the audited twenty-frame entry")
        expected = (16, 1) if variant == "base" else (48, 3)
        if (self.model.config.n_tokens, self.model.num_quantizers) != expected:
            raise ValueError("Released ActionCodec token/quantizer architecture mismatch")
        self.quantizers = expected[1] if quantizers is None else quantizers
        if type(self.quantizers) is not int or not 1 <= self.quantizers <= expected[1]:
            raise ValueError("Invalid released RVQ prefix depth")
        self.token_count = self.quantizers * 16
        # Exact model-card normalizer, not affine q01/q99 scaling.
        self.scale = np.array(
            [0.9375, 0.91071427, 0.9375, 0.20357142, 0.26357144, 0.375], np.float32
        )

    def normalized(self, actions):
        value = actions_checked(actions).copy()
        value[:, :6] /= self.scale
        value[:, 6] = -value[:, 6]
        return np.clip(value, -1, 1)

    def encode(self, actions, context):
        value = self.normalized(actions)
        if len(value) % 20:
            raise ValueError("ActionCodec needs whole native twenty-frame blocks")
        tokens = [
            self.model.encode(block[None], embodiment_ids=self.embodiment)[0]
            for block in np.split(value, len(value) // 20)
        ]
        return Packet(np.array(tokens, dtype=np.int64)[:, : self.token_count], 2048, len(value))

    def decode(self, packet, context):
        if (
            packet.vocabulary != 2048
            or packet.horizon % 20
            or packet.ids.shape != (packet.horizon // 20, self.token_count)
        ):
            raise ValueError("ActionCodec native token/block shape mismatch")
        decoded = []
        for tokens in packet.ids:
            block, mask = self.model.decode(
                tokens[None], embodiment_ids=self.embodiment, durations=[1.0]
            )
            if block.shape != (1, 20, 7) or mask.shape != (1, 20) or not np.all(mask == 1):
                raise ValueError("Native decode introduced unexpected padding or action shape")
            decoded.append(block[0])
        value = np.concatenate(decoded).astype(np.float32)
        value[:, :6] *= self.scale
        value[:, 6] *= -1
        return value

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self):
        return {
            "codec": "actioncodec" if self.variant == "base" else "actioncodec_rvq",
            "assets": self.assets,
            "precision": "float32",
            "device": self.device.type,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "native_horizon": 20,
            "tokens_per_native_block": self.token_count,
            "quantizers": self.quantizers,
            "encoded_quantizers": self.model.num_quantizers,
            "prefix_scope": "native layer-major RVQ prefix; all released encode work included",
            "embodiment": "a_franka_libero_20hz",
            "physical_rate_conversion": "none",
            "normalization": "released model-card symmetric quantile maxima; signed open-gripper",
            "scale": self.scale.tolist(),
            "side_information": "horizon and fixed embodiment only",
            "pretraining_exposure": "released Base explicitly includes LIBERO",
        }

    def close(self):
        self.model = None
