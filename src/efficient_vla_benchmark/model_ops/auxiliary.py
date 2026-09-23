"""One bounded, serialized auxiliary CUDA pass with explicit completion accounting."""

import time
from concurrent.futures import ThreadPoolExecutor

import torch


class AuxiliaryCUDARunner:
    def __init__(self, device):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evb-auxiliary")
        self.future = None
        self.closed = False

    def submit(self, work):
        if self.closed or self.future is not None:
            raise RuntimeError("Auxiliary work must complete before another pass is submitted")
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))

        def run():
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            started = time.perf_counter_ns()
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                # A discarded stochastic action head must not advance the next served noise.
                # A backend drains before its next forward/reset; one policy owns this process.
                with torch.inference_mode(), torch.random.fork_rng(devices=[index]):
                    try:
                        work()
                    finally:
                        self.stream.synchronize()
            return {"auxiliary_forward_ms": (time.perf_counter_ns() - started) / 1e6}

        self.future = self.executor.submit(run)

    def done(self):
        return self.future is not None and self.future.done()

    def drain(self):
        if self.future is None:
            return None
        future, self.future = self.future, None
        started = time.perf_counter_ns()
        result = future.result()
        result["auxiliary_join_wait_ms"] = (time.perf_counter_ns() - started) / 1e6
        return result

    def close(self):
        if self.closed:
            return
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True)
            self.closed = True
