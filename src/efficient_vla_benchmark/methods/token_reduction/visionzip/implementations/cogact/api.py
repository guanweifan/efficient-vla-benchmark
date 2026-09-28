def build(backend, config, resources):
    from .method import VisionZip

    return VisionZip(backend, config)
