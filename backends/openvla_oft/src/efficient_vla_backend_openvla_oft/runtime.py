"""Device-specific monitoring behind the common profiling contract."""

import torch

from efficient_vla_benchmark.core.provenance import driver_version


class TorchCUDAMonitor:
    def __init__(self, device, drain_auxiliary=None):
        self.device = device
        self.drain_auxiliary = drain_auxiliary

    def synchronize(self):
        if self.drain_auxiliary is not None:
            self.drain_auxiliary()
        torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        return {
            "allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }

    def hardware(self):
        return {
            "gpu": torch.cuda.get_device_name(self.device),
            "torch_cuda": torch.version.cuda,
            "driver_version": driver_version(),
            "cpu_threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "runtime": "pytorch_cuda",
        }
