from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


def test_class_and_instance_forward_restore_and_idempotence():
    class Module:
        def forward(self, value):
            return value + 1

    a, b = Module(), Module()
    b.forward = lambda value: value * 2
    patches = ForwardPatches()
    for module in (a, b):
        patches.wrap(module, lambda original: lambda value: original(value) + 10)
    assert a.forward(2) == 13 and b.forward(2) == 14
    patches.close()
    patches.close()
    assert a.forward(2) == 3 and b.forward(2) == 4
    assert "forward" not in a.__dict__


def test_named_sampler_method_restored():
    class Sampler:
        def sample(self):
            return 3

    sampler = Sampler()
    patches = ForwardPatches()
    patches.wrap(sampler, lambda original: lambda: original() + 2, "sample")
    assert sampler.sample() == 5
    patches.close()
    assert sampler.sample() == 3 and "sample" not in sampler.__dict__
