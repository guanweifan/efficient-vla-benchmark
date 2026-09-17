"""Lazy backend factory."""


def build(backend, config, resources):
    from .method import VLAIAP

    return VLAIAP(backend, config)
