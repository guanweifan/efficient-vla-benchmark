"""Static non-contiguous layer pruning; independent of token/DiT cores."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LayerPruningConfig:
    enabled: bool = True
    prune_layers: int = 4
    calibration_samples: int = 32
    calibration_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown EfficientVLA layer-pruning parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.prune_layers) is not int or config.prune_layers < 0:
            raise ValueError("prune_layers must be a nonnegative integer")
        if (
            type(config.calibration_samples) is not int
            or config.calibration_samples < 4
            or config.calibration_samples % 4
        ):
            raise ValueError("calibration_samples must be a positive multiple of four")
        if type(config.calibration_seed) is not int or config.calibration_seed < 0:
            raise ValueError("calibration_seed must be nonnegative")
        return config

    def calibration_parameters(self):
        return {"samples": self.calibration_samples, "seed": self.calibration_seed}
