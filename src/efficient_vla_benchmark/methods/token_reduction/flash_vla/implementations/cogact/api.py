"""CogACT native-attention binding for the independent ICS contribution."""


def build(backend, config, resources):
    from .method import FlashVLA

    return FlashVLA(backend, config)
