def build(backend, config, resources):
    from .method import FastV

    return FastV(backend, config)
