def build(backend, config, resources):
    from .method import ILSA

    return ILSA(backend, config)
