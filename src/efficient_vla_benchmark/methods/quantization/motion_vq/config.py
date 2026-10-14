import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class MotionVQConfig:
    threshold: float = 0.8
    calibration_samples: int = 40
    calibration_seed: int = 0
    calibration_tokens: int = 256
    kmeans_iterations: int = 20
    block_batch_size: int = 4
    hessian_damping: float = 0.01
    kmeans_seed: int = 42

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown MotionVQ parameter")
        c = cls(**parameters)
        for key in ("threshold", "hessian_damping"):
            v = getattr(c, key)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError("Invalid MotionVQ scalar")
        for key in (
            "calibration_samples",
            "calibration_seed",
            "calibration_tokens",
            "kmeans_iterations",
            "block_batch_size",
            "kmeans_seed",
        ):
            if type(getattr(c, key)) is not int:
                raise ValueError("MotionVQ counts must be integers")
        if (
            not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
            or c.calibration_seed < 0
            or c.kmeans_seed < 0
            or not 16 <= c.calibration_tokens <= 1024
            or not 1 <= c.kmeans_iterations <= 100
            or not 1 <= c.block_batch_size <= 16
        ):
            raise ValueError("Invalid MotionVQ calibration geometry")
        return c

    def calibration_parameters(self):
        values = asdict(self)
        values.pop("threshold")
        return values
