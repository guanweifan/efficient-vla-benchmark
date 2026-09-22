"""Lazy native-cognition binding for SAFE-Pruner."""


def preflight(config, resources):
    if max(config.pruning_layer, *config.reference_layers) >= 32:
        raise ValueError("CogACT SAFE-Pruner layer indices must be below 32")


def build(backend, config, resources):
    from .method import SAFEPruner

    return SAFEPruner(backend, config)
