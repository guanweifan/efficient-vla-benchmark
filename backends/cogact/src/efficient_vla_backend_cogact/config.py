"""Explicit native inference settings, independent of model dependencies."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CogACTParameters:
    num_ddim_steps: int = 10
    cfg_scale: float = 1.5

    @classmethod
    def from_mapping(cls, values):
        unknown = set(values) - {"num_ddim_steps", "cfg_scale"}
        if unknown:
            raise ValueError("Unknown CogACT backend parameters")
        result = cls(**values)
        if type(result.num_ddim_steps) is not int or not 1 <= result.num_ddim_steps <= 1000:
            raise ValueError("DDIM steps must be an integer from 1 to 1000")
        if (
            type(result.cfg_scale) not in {int, float}
            or not math.isfinite(result.cfg_scale)
            or result.cfg_scale < 1.0
        ):
            raise ValueError("CFG scale must be finite and at least one")
        return result


def validate(config):
    if config.model_variant != "libero_joint":
        raise ValueError("CogACT currently supports the joint-four-suite LIBERO checkpoint")
    if config.precision not in {"auto", "bfloat16"}:
        raise ValueError("CogACT uses a BF16 VLM and native FP32 diffusion head")
    if not 1 <= config.execution_horizon <= 16:
        raise ValueError("This CogACT variant predicts sixteen actions per query")
    CogACTParameters.from_mapping(config.backend_parameters)
