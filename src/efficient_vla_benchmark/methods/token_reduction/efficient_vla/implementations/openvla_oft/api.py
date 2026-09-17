def build(backend, config, resources):
    from .method import EfficientVLA

    return EfficientVLA(backend, config)
