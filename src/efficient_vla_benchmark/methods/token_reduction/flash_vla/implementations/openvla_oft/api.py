def build(backend, config, resources):
    from .method import FlashVLA

    return FlashVLA(backend, config)
