"""CogACT device monitoring without coupling to another backend."""

import torch

from efficient_vla_benchmark.core.provenance import driver_version


class TorchCUDAMonitor:
    def __init__(self, device):
        self.device = device

    def synchronize(self):
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
