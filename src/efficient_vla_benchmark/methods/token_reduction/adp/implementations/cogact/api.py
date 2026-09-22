"""Single-camera CogACT binding, with explicit observed-state requirements."""


def preflight(config, resources):
    if len(config.view_weights) != 1:
        raise ValueError("CogACT ADP requires exactly one camera weight; use [1.0]")


def build(backend, config, resources):
    from .method import ADP

    return ADP(backend, config)
