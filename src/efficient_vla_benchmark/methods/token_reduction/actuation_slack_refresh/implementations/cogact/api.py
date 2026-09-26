def build(backend, config, resources):
    from .method import ActuationSlackRefresh

    return ActuationSlackRefresh(backend, config)


def build_reuse(backend, config, resources):
    from .reuse import ActuationSlackReuse

    return ActuationSlackReuse(backend, config)
