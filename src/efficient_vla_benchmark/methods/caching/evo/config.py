from dataclasses import dataclass


@dataclass(frozen=True)
class EVOConfig:
    enabled: bool = True
    refresh_budget: int = 120

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {"enabled", "refresh_budget"}:
            raise ValueError("Unknown EVO parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool or type(config.refresh_budget) is not int:
            raise ValueError("EVO requires boolean enabled and integer refresh_budget")
        if config.refresh_budget < 1:
            raise ValueError("EVO requires a positive global refresh budget")
        return config
