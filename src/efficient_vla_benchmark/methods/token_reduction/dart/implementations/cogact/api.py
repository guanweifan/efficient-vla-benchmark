def build(backend, config, resources):
    from .method import DART

    return DART(backend, config)
