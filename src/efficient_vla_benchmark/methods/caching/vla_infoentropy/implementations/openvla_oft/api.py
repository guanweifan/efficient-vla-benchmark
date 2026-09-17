def build(backend, config, resources):
    from .method import VLAInfoEntropy

    return VLAInfoEntropy(backend, config)
