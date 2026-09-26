"""Reversible instance-forward replacement, including direct forward calls."""


class ForwardPatches:
    def __init__(self):
        self.saved = []

    def wrap(self, module, factory):
        previous = module.__dict__.get("forward")
        had_instance = "forward" in module.__dict__
        replacement = factory(module.forward)
        self.saved.append((module, had_instance, previous))
        module.forward = replacement

    def close(self):
        for module, had_instance, previous in reversed(self.saved):
            if had_instance:
                module.forward = previous
            else:
                del module.forward
        self.saved.clear()
