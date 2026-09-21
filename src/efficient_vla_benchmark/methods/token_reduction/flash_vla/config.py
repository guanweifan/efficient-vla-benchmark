from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FlashVLAConfig:
    enabled: bool = True
    pruning_layer: int = 1
    tokens_per_view: int = 160
    svd_driver: str = "auto"

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown FlashVLA ICS parameter; action reuse is not implemented")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.pruning_layer) is not int or not 0 <= config.pruning_layer < 31:
            raise ValueError("pruning_layer must be in [0, 30]")
        if type(config.tokens_per_view) is not int or not 1 <= config.tokens_per_view <= 256:
            raise ValueError("tokens_per_view must be in [1, 256]")
        if config.svd_driver not in ("auto", "gesvd", "gesvdj"):
            raise ValueError("svd_driver must be auto, gesvd or gesvdj; no approximate SVD")
        return config
