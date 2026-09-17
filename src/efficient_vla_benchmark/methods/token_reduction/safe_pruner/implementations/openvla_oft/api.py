"""Lazy factory; reject unsupported layers before loading a checkpoint."""


def preflight(config, resources):
    if max(config.pruning_layer, *config.reference_layers) >= 32:
        raise ValueError("OFT SAFE-Pruner layer indices must be below 32")


def build(backend, config, resources):
    from .method import SAFEPruner

    return SAFEPruner(backend, config)
