def build(backend, config, resources):
    from .method import ActuationSlackRefresh

    return ActuationSlackRefresh(backend, config)
