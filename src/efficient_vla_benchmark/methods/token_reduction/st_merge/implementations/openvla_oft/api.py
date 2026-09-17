def build(backend, config, resources):
    from .method import STMerge

    return STMerge(backend, config)
