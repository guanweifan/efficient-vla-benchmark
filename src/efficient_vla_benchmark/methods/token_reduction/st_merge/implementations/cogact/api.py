"""Lazy factory for spatial merging on native CogACT visual encoders."""


def build(backend, config, resources):
    from .method import STMerge

    return STMerge(backend, config)
