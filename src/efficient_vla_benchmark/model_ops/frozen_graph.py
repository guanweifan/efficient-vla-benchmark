"""Bounded fixed-geometry CUDA graphs for deterministic frozen tensor operators."""

import time

import torch


class GraphPool:
    """Sequential shared scratch pool; no cached policy output or global state."""

    def __init__(self):
        self.device = self.stream = self.pool = None

    def prepare(self, device):
        if self.device is None:
            self.device = device
            with torch.cuda.device(device):
                self.stream = torch.cuda.Stream(device=device)
                self.pool = torch.cuda.graph_pool_handle()
        elif self.device != device:
            raise ValueError("A graph pool cannot span CUDA devices")


class FrozenOpGraph:
    """Exactly one graph for one input geometry; dynamic tensor values are copied.

    Only use with deterministic, stream-correct, inference-only operations. Every
    input is overwritten before replay, and outputs are cloned before another
    graph uses the shared scratch pool. Timestep-dependent scales are inputs,
    not a Python value frozen accidentally during capture. No eager fallback.
    The owner clears graphs before moving/reloading weights or changing geometry.
    """

    def __init__(self, operation, pool):
        self.operation, self.pool = operation, pool
        self.clear()

    def clear(self):
        self.graph = self.inputs = self.output = self.signature = None
        self.replays = 0
        self.capture_seconds = 0.0

    @torch.no_grad()
    def __call__(self, *inputs):
        if not inputs or any(
            not isinstance(x, torch.Tensor)
            or not x.is_cuda
            or not x.numel()
            or x.requires_grad
            or x.device != inputs[0].device
            for x in inputs
        ):
            raise ValueError("Frozen graphs require nonempty colocated inference CUDA tensors")
        signature = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs)
        if self.signature is not None and self.signature != signature:
            raise ValueError("Frozen graph geometry changed; clear/rebuild explicitly")
        with torch.cuda.device(inputs[0].device):
            if self.graph is None:
                start = time.perf_counter()
                self.pool.prepare(inputs[0].device)
                static = tuple(x.contiguous().clone() for x in inputs)
                stream = self.pool.stream
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.operation(*static)
                    self.operation(*static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=self.pool.pool):
                    output = self.operation(*static)
                if not isinstance(output, torch.Tensor):
                    raise ValueError("Frozen graph operation must return one tensor")
                self.graph, self.inputs, self.output = graph, static, output
                self.signature = signature
                self.capture_seconds += time.perf_counter() - start
            for target, current in zip(self.inputs, inputs):
                target.copy_(current)
            self.graph.replay()
            self.replays += 1
            return self.output.clone()

    def metadata(self):
        return dict(
            captured=self.graph is not None,
            replays=self.replays,
            capture_seconds=self.capture_seconds,
            input_copies_and_output_clone_timed=True,
            cached_policy_outputs=False,
        )
