# Packed quantization operations

Optional Torch/Triton operations, independent of methods, backends and calibration.
These are feasibility primitives, not a selectable quantization algorithm.

- `packed.py`: signed two's-complement 2/4/8-bit row packing into uint8.
  `unpack_signed` is a diagnostic oracle, never the production forward path.
- `linear.py`: converts integer codes once, owns packed buffers/scales and exposes
  `execution_metadata()`. Same-topology `state_dict` reload is supported. Constructor
  geometry, bit width and activation-scale policy are part of the caller's artifact
  identity, not inferred from raw tensor names.
- `kernel.py`: tile-local unpack/dequantization inside GEMM; no persistent or
  per-query full floating weight reconstruction. Floating products accumulate in
  FP32; this is **not** an integer-MMA W4A8 kernel. No silent CPU fallback.

The logical packed-weight shape is `(output, packed_input)`, with output-contiguous
transposed strides for coalesced N-lane loads. This is one physical packed buffer,
not a second copy. Packing codes, scale rounding, K reduction and native precision
are unchanged; GPU tests compare the previous row-major path exactly. Old ad-hoc
assumptions of contiguous packed storage are invalid; use tensor strides or the
diagnostic `unpack_signed` function.

Weights have per-row or per-row/K-group scales. Optional A2/A4/A8 activations are
stored as INT8 codes; A16 bypasses activation quantization without changing the
native floating dtype. The caller declares symmetric or two's-complement bounds.
Per-input-channel scales remain **inside** the K reduction. Replacing these
scales with a group maximum changes the quantizer and is not allowed. Bias is
added before output rounding; methods with output transforms own their later bias.
FP16/BF16/FP32 output precision is retained. The explicit scale-rounding flag
distinguishes float-scale weight-only quantization from the calibrated A8 path.
Long half-precision inputs use wider M/N tiles while retaining the same K=32
reduction. Short inputs and FP32 keep compact tiles; kernel/reference checks must
establish parity before reusing success evidence across an execution-only change.

Callers own target selection, calibration, checkpoint/artifact verification,
conversion lifetime and native precision. They must supply colocated tensors and
use a fresh native model to undo irreversible conversion. Do not compose this
implementation with AWQ or silently replace its author kernel.

Tests cover packing tails, signed extremes, group tails, real GPU execution,
activation scales, bias, all three precisions, reload, empty batches and rejection
of CPU execution. Run explicitly in an authorized backend runtime:

```bash
python -m pytest src/efficient_vla_benchmark/model_ops/quantization/tests
```

Full method admission additionally requires checkpoint-matched calibration,
whole-policy reference checks, native protocol smoke and full-cost replay timing.
Packing efficiency alone is neither a latency claim nor evidence of task accuracy.
