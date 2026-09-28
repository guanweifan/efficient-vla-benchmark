def build(backend, config, resources):
    from .method import AMMRCurrentFrame

    return AMMRCurrentFrame(backend, config)
