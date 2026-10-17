"""One W4 buffer with runtime A2/A4/A8/native dispatch; never reconstruct full weights."""

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear


class DynamicW4Linear(PackedLinear):
    def __init__(self, entry):
        super().__init__(
            entry["codes"],
            entry["weight_scale"],
            bits=4,
            bias=entry["bias"],
            activation_amax=entry["activation_amax"],
            activation_bits=16,
            round_weight_scale=True,
        )
        self.set_active_bits(16)

    def set_active_bits(self, bits):
        if type(bits) is not int or bits not in (2, 4, 8, 16):
            raise ValueError("Invalid DyQ activation precision")
        self.activation_bits = bits
        self.activation_qmin = -(1 << (min(bits, 8) - 1))


def execution_runtime(config, path):
    if config.runtime == "native_float":
        return None
    from efficient_vla_benchmark.model_ops.quantization.mixed_runtime import MixedW4Runtime

    return MixedW4Runtime(path)


def make_linear(entry, config, runtime):
    if config.runtime == "native_float":
        return DynamicW4Linear(entry)
    from efficient_vla_benchmark.model_ops.quantization.mixed_linear import MixedPrecisionW4Linear

    if entry["native_dtype"] != "torch.bfloat16" or entry["bias"] is not None:
        raise ValueError("DyQ mixed integer runtime requires bias-free BF16 decoder projections")
    return MixedPrecisionW4Linear(entry["codes"], entry["weight_scale"], runtime=runtime)
