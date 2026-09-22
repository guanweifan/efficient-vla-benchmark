"""Lazy factory for native causal information-entropy KV reuse."""


def build(backend, config, resources):
    from .method import VLAInfoEntropy

    return VLAInfoEntropy(backend, config)
