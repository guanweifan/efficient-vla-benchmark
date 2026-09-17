"""OFT compatibility checks and one complete SpecPrune factory."""


def preflight(config, resources):
    for name in ("importance_layers", "pruning_layers", "global_layers"):
        if max(getattr(config, name)) >= 32:
            raise ValueError("OFT SpecPrune layer indices must be below 32")
    for name in (
        "primary_static_budget",
        "wrist_static_budget",
        "primary_precise_static_budget",
        "wrist_precise_static_budget",
    ):
        if getattr(config, name) > 256:
            raise ValueError("OFT provides 256 visual tokens per view")
    if config.min_visual_tokens > 512:
        raise ValueError("OFT provides 512 visual tokens in total")


def build(backend, config, resources):
    from .method import SpecPrune

    return SpecPrune(backend, config)
