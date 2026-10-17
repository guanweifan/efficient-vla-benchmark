"""Real packed W4/A4 execution of the explicitly scoped official-code variant."""

import torch
from torch import nn

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear
from efficient_vla_benchmark.model_ops.quantization.native_fp32 import PackedFP32Linear

from .rotation import InputRotation, rotate


class HoloQLinear(nn.Module):
    def __init__(self, entry, step_provider, *, runtime="native_float"):
        super().__init__()
        self.graph = None
        if runtime not in ("native_float", "tinychat_fp16"):
            raise ValueError("Unsupported HoloQ runtime")
        dtype = getattr(torch, entry["native_dtype"].split(".")[-1])
        self.input_rotation = InputRotation(entry["transform"], dtype)
        self.register_buffer(
            "output_rotation",
            None
            if entry["transform"]["output"] is None
            else entry["transform"]["output"].to(dtype),
        )
        self.register_buffer("scale_table", entry["activation_steps"].to(dtype))
        self.register_buffer("bias", None if entry["bias"] is None else entry["bias"].to(dtype))
        self._scale_table_ref = None
        self._scale_rows = ()
        self.step_provider = step_provider
        self.is_dit = entry["side"] != "language"
        options = dict(
            group_size=entry["weight_group_size"],
            activation_bits=4,
            activation_step=self.scale_table[0],
            activation_qmin=-8,
        )
        self.fused_preparation = self.fused_output = None
        if runtime == "tinychat_fp16":
            from efficient_vla_benchmark.model_ops.quantization.channel_linear import (
                ChannelPackedLinear,
            )
            from efficient_vla_benchmark.model_ops.quantization.rotated_channel import (
                RotatedChannelPreparation,
                RotatedOutput,
            )

            self.linear = ChannelPackedLinear(entry["codes"], entry["weight_scales"], **options)
            self.fused_preparation = RotatedChannelPreparation(4, qmin=-8)
            self.fused_output = RotatedOutput() if self.output_rotation is not None else None
        else:
            linear_type = (
                PackedFP32Linear if self.is_dit and dtype == torch.float32 else PackedLinear
            )
            self.linear = linear_type(
                entry["codes"], entry["weight_scales"], bits=4, round_weight_scale=False, **options
            )
        # scale_table is the sole serialized authority. The selected row is a
        # transient view; loading a separately saved row into that view would
        # overwrite whichever timestep was last used before load_state_dict().
        self.linear.register_buffer(
            "activation_step", self.linear.activation_step, persistent=False
        )
        self.in_features, self.out_features = self.linear.in_features, self.linear.out_features

    def _clear_scale_views(self):
        self._scale_table_ref = None
        self._scale_rows = ()

    def _apply(self, fn, recurse=True):
        # Ordinary Python views are not buffers: release them before a move.
        self.clear_graph()
        self._clear_scale_views()
        if self.fused_preparation is not None:
            self.fused_preparation.clear()
        if self.fused_output is not None:
            self.fused_output.clear()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_graph()
        self._clear_scale_views()
        return super()._load_from_state_dict(*args, **kwargs)

    def configure_dit_graph(self, engine, pool):
        if not self.is_dit or self.fused_preparation is None:
            raise ValueError("Only fixed-shape FP16 packed DiT projections support graphs")
        from efficient_vla_benchmark.model_ops.frozen_graph import FrozenOpGraph

        self.linear.kernel.stream_runtime = engine
        self.graph = FrozenOpGraph(self._execute, pool)

    def clear_graph(self):
        if self.graph is not None:
            self.graph.clear()

    def _scale(self, row):
        # In-place value updates remain visible through views; replacement
        # of the registered table rebuilds them rather than using old storage.
        if self._scale_table_ref is not self.scale_table:
            self._scale_rows = self.scale_table.unbind(0)
            self._scale_table_ref = self.scale_table
        return self._scale_rows[row]

    def forward(self, inputs):
        row = 0
        if self.is_dit:
            row = self.step_provider()
            if type(row) is not int or not 0 <= row < len(self.scale_table):
                raise ValueError("Missing or invalid native denoising timestep")
        scale = self._scale(row)
        if self.graph is not None:
            replay = self.graph.replays > 0
            result = self.graph(inputs, scale)
            self.linear.kernel_invocations += int(replay)
            return result
        return self._execute(inputs, scale)

    def _execute(self, inputs, scale):
        if self.fused_preparation is not None:
            _, prepared = self.fused_preparation(
                inputs, self.input_rotation.blocks, scale, self.input_rotation.permutation
            )
            if not prepared.numel():
                return inputs.new_empty((*inputs.shape[:-1], self.out_features))
            output = self.linear.kernel(prepared)
            self.linear.kernel_invocations += 1
            if self.fused_output is not None:
                return self.fused_output(output, self.output_rotation, self.bias)
            output = output.to(inputs.dtype)
            return output if self.bias is None else output + self.bias.to(output.dtype)
        self.linear.activation_step = scale
        x = self.input_rotation(inputs)
        output = self.linear(x)
        if self.output_rotation is not None:
            output = rotate(output, self.output_rotation)
        return output if self.bias is None else output + self.bias
