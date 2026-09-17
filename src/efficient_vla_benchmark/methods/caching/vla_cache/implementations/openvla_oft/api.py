"""OFT compatibility checks and one complete VLA-Cache factory."""


def preflight(config, resources):
    if config.task_attention_layer >= 32 or max(config.reuse_layers) >= 32:
        raise ValueError("OFT VLA-Cache layer indices must be below 32")
    if max(config.static_budget_per_view, config.protected_tokens_per_view) > 256:
        raise ValueError("OFT provides 256 visual tokens per view")


def build(backend, config, resources):
    from .method import VLACache

    return VLACache(backend, config)
