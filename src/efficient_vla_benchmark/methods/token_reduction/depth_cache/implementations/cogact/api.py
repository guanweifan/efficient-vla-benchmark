def build(backend, config, resources):
    from .method import DepthCachePrimary

    return DepthCachePrimary(backend, config)
