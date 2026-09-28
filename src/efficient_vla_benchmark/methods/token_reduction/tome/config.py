"""ToMe's constant per-block removal count, not a retention ratio."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ToMeConfig:
    enabled: bool = True
    tokens_per_layer: int = 4

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown ToMe parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.tokens_per_layer) is not int or not 0 <= config.tokens_per_layer <= 128:
            raise ValueError("tokens_per_layer must be an integer in [0, 128]")
        return config
