def build(backend, config, resources):
    from .method import DivPrune

    return DivPrune(backend, config)
