"""Lazy method construction and model bounds."""


def preflight(config, resources):
    if max(config.pruning_layer, config.history_layer) >= 32:
        raise ValueError("OFT layer indices must be below 32")


def build(backend, config, resources):
    from .method import VLAPruner

    return VLAPruner(backend, config)
