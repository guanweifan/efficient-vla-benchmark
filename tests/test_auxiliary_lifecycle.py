import pytest

from efficient_vla_benchmark.core.lifecycle import MethodHost
from efficient_vla_benchmark.core.method import InferenceMethod


class DeferredRunner:
    def __init__(self):
        self.work = None
        self.closed = False

    def submit(self, work):
        assert self.work is None
        self.work = work

    def done(self):
        return False

    def drain(self):
        if self.work is None:
            return None
        work, self.work = self.work, None
        work()
        return {"auxiliary_forward_ms": 10.0, "auxiliary_join_wait_ms": 1.0}

    def close(self):
        self.drain()
        self.closed = True


class Host(MethodHost):
    def __init__(self):
        self.initialize_method_state()

    def make_auxiliary_runner(self):
        return DeferredRunner()


class RefreshMethod(InferenceMethod):
    def __init__(self):
        self.auxiliary = False
        self.primary_outputs = []

    def wants_auxiliary_query(self):
        return True

    def begin_auxiliary_query(self):
        self.auxiliary = True

    def end_auxiliary_query(self):
        self.auxiliary = False

    def after_query(self, output):
        self.primary_outputs.append(output)


def test_auxiliary_output_never_served_and_same_prepared_inputs_reused():
    host, method = Host(), RefreshMethod()
    host.attach_method(method)
    prepared, calls = {"frame": object()}, []

    def native(value):
        assert value is prepared
        calls.append(method.auxiliary)
        return "discarded" if method.auxiliary else "served"

    assert host.run_native_query(prepared, native) == "served"
    assert calls == [False]
    host.drain_auxiliary()
    assert calls == [False, True]
    assert method.primary_outputs == ["served"]
    assert host.query_diagnostics["auxiliary_queries"] == 1
    assert host.query_diagnostics["auxiliary_forward_ms"] == 10.0
    host.detach_method()
    assert host.auxiliary_runner is None
    host.close()
    host.close()


def test_auxiliary_error_is_propagated_and_mode_restored():
    host, method = Host(), RefreshMethod()
    host.attach_method(method)

    def native(value):
        if method.auxiliary:
            raise RuntimeError("auxiliary failure")
        return 1

    assert host.run_native_query({}, native) == 1
    with pytest.raises(RuntimeError, match="auxiliary failure"):
        host.drain_auxiliary()
    assert not method.auxiliary
    host.close()
