"""Explicit OFT-only compatibility for ADP's released layer-zero scorer."""


def preflight(config, resources):
    if len(config.view_weights) != 2:
        raise ValueError("OFT ADP requires two view weights")


def build(backend, config, resources):
    from .method import ADP

    return ADP(backend, config)
