def build(backend, config, resources):
    from ... import method

    return method.SPVLAScheduling(config)
