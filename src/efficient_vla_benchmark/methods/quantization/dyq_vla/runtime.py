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
