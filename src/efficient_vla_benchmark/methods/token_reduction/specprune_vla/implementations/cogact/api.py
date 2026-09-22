"""Validate the native single-view backbone before expensive loading."""


def preflight(config, resources):
    for name in ("importance_layers", "pruning_layers", "global_layers"):
        if max(getattr(config, name)) >= 32:
            raise ValueError("CogACT SpecPrune layer indices must be below 32")
    if (
        max(
            config.primary_static_budget,
            config.primary_precise_static_budget,
            config.min_visual_tokens,
        )
        > 256
    ):
        raise ValueError("CogACT provides 256 visual tokens in total")


def build(backend, config, resources):
    from .method import SpecPrune

    return SpecPrune(backend, config)
