def build(backend, config, resources):
    from .method import ProbeFlow

    return ProbeFlow(backend, config)
