"""Reversible instance-forward replacement, including direct forward calls."""


class ForwardPatches:
    def __init__(self):
        self.saved = []

    def wrap(self, module, factory, attribute="forward"):
        previous = module.__dict__.get(attribute)
        had_instance = attribute in module.__dict__
        replacement = factory(getattr(module, attribute))
        self.saved.append((module, attribute, had_instance, previous))
        setattr(module, attribute, replacement)

    def close(self):
        for module, attribute, had_instance, previous in reversed(self.saved):
            if had_instance:
                setattr(module, attribute, previous)
            else:
                delattr(module, attribute)
        self.saved.clear()
