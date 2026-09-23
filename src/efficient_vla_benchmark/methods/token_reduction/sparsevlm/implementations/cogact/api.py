def build(backend, config, resources):
    from .method import SparseVLM

    return SparseVLM(backend, config)
