"""Lazy backend factory."""


def preflight(config, resources):
    if config.merging_layer >= 32:
        raise ValueError("OFT merging_layer must be below 32")


def build(backend, config, resources):
    from .method import TEAMVLA

    return TEAMVLA(backend, config)
