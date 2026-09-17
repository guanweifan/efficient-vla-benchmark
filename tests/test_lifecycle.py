import pytest

from efficient_vla_benchmark.core.lifecycle import MethodHost
from efficient_vla_benchmark.core.method import InferenceMethod


def test_single_method_and_irreversible_conversion():
    host = MethodHost()
    host.initialize_method_state()
    method = InferenceMethod()
    method.name, method.reversible = "converted", False
    host.attach_method(method)
    with pytest.raises(ValueError, match="Exactly one"):
        host.attach_method(InferenceMethod())
    with pytest.raises(ValueError, match="fresh backend"):
        host.detach_method()
    host.close()
    host.close()
    with pytest.raises(RuntimeError, match="closed"):
        host.observe(None)


def test_state_is_instance_local():
    first, second = MethodHost(), MethodHost()
    first.initialize_method_state()
    second.initialize_method_state()
    first.attach_method(InferenceMethod())
    assert second.method is None
    first.detach_method()
    assert first.method is None
