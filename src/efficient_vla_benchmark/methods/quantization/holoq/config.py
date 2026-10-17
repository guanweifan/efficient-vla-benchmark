"""Fixed W4A4 official-code variant on a native CogACT sampler."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class HoloQConfig:
    runtime: str = "native_float"
    dit_graphs: bool = False
    block_size: int = 64
    gptq_block_size: int = 128
    gptq_damp_percent: float = 0.01
    activation_percentile: float = 99.9
    calibration_samples: int = 40
    calibration_tokens: int = 256
    calibration_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown HoloQ parameter")
        c = cls(**parameters)
        if c.runtime not in ("native_float", "tinychat_fp16"):
            raise ValueError("Unsupported HoloQ runtime")
        if type(c.dit_graphs) is not bool or (c.dit_graphs and c.runtime != "tinychat_fp16"):
            raise ValueError("DiT graphs require explicit FP16 packed execution")
        if type(c.block_size) is not int or c.block_size not in (16, 32, 64, 128):
            raise ValueError("Unsupported composite rotation block size")
        if type(c.gptq_block_size) is not int or c.gptq_block_size not in (64, 128, 256):
            raise ValueError("Unsupported GPTQ block size")
        if c.runtime == "tinychat_fp16" and c.gptq_block_size < 128:
            raise ValueError("HoloQ FP16 packed execution requires GPTQ blocks of at least 128")
        if (
            type(c.calibration_samples) is not int
            or not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
        ):
            raise ValueError("Calibration count must be a multiple of four")
        if type(c.calibration_tokens) is not int or not 16 <= c.calibration_tokens <= 1024:
            raise ValueError("Invalid captured token limit")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid calibration seed")
        for key in ("gptq_damp_percent", "activation_percentile"):
            value = getattr(c, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Calibration scalar must be finite")
        if not 0 < c.gptq_damp_percent <= 1 or not 0 < c.activation_percentile <= 100:
            raise ValueError("Invalid calibration scalar")
        return c

    def calibration_parameters(self):
        result = asdict(self)
        # Only native policies run during calibration; preserve the same codes,
        # transforms and complete scale tables across explicit execution engines.
        result.pop("runtime")
        result.pop("dit_graphs")
        return result
