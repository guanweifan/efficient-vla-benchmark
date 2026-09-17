from efficient_vla_benchmark.core.method import InferenceMethod


class ExampleScale(InferenceMethod):
    name = "example_scale"

    def __init__(self, backend, config):
        self.backend = backend
        self.previous = backend.action_scale
        backend.action_scale = config.scale

    def close(self):
        self.backend.action_scale = self.previous


def build(backend, config, resources):
    return ExampleScale(backend, config)
