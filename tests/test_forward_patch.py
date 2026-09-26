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
