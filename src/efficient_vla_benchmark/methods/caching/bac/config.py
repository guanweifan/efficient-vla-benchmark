"""Training-free BAC schedule budget and calibration parameters."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class BACConfig:
    enabled: bool = True
    num_updates: int = 6
    protected_blocks: int = 5
    calibration_samples: int = 64
    calibration_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown BAC parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("num_updates", "protected_blocks", "calibration_samples", "calibration_seed"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if config.num_updates < 1 or config.protected_blocks < 1:
            raise ValueError("ACS updates and BUA protected blocks must be positive")
        if config.calibration_samples < 4 or config.calibration_samples % 4:
            raise ValueError("Calibration must be balanced across four suites")
        if not 0 <= config.calibration_seed < 2**32:
            raise ValueError("Invalid calibration seed")
        return config

    def calibration_parameters(self):
        return {"samples": self.calibration_samples, "seed": self.calibration_seed}
