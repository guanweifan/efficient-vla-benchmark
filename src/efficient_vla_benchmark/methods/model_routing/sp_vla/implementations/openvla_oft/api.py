def build(backend, config, resources):
    if config.enabled and config.token_pruning:
        from .method import SPVLA

        return SPVLA(backend, config)
    from ... import method

    return method.SPVLAScheduling(config)
