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
- `native_fp32.py`: explicit packed storage with per-call temporary FP32 decode
  followed by native GEMM. Use only when native reduction order is required;
  count the decode time and full temporary workspace. No persistent weight cache
  or automatic fallback is allowed. Existing tile execution is unchanged.
- `tinychat.py`: caller-supplied symmetric W4g128 codes packed for the pinned
  TinyChat CUDA GEMM/GEMV. This reuses execution kernels, NOT AWQ calibration.
  FP16 scale storage/products/accumulation and BF16 boundary casts are explicit.
  It does not accept native-FP32 modules or promise integer MMA. No existing
  method changes its selected engine implicitly.
  For 8–32 rows, pad to 33 independent rows and slice the result to select the
  author's single-split kernel: its short-row split-K path uses uninitialized
  semaphores and can depend on recycled allocator contents. The copy is timed,
  and changed reduction order needs fresh policy validation for affected shapes.
  Longer matrices and the fewer-than-eight-row GEMV path are unchanged.
- `gptq.py`: full-Hessian, cross-block error compensation, optionally dynamic
  input groups. Group scales follow the author's corrected global-weight rule;
  no activation ordering, RTN fallback or policy-target selection is implied.
- `integer_linear.py` / `integer_kernel.py`: explicit dynamic per-token A8 and
  per-output-channel weight scales, with signed INT8 Tensor Core multiplication,
  INT32 accumulation and a fused FP32 scale/bias epilogue. W4 storage is expanded
  to INT8 in registers; it is **not** INT4 MMA. Activation quantization is timed.
  This path cannot replace a per-input-channel activation quantizer without
  changing its mathematics. SmoothQuant validates this engine end to end; other
  quantizers do not inherit that validation or switch to it implicitly.
- `channel_prepare.py`: fused per-input-channel quantization, preserving native
  intermediate rounding, plus explicit FP16 reconstruction for a packed floating
  kernel. It returns INT8 codes and is not integer MMA. Caller-owned frozen scales
  remain inside the quantizer; no replacement by per-token/group scales is allowed.
- `channel_linear.py`: explicitly selected TinyChat W4 execution after that native
  channel quantizer. It accepts native FP32 boundaries but converts packed GEMM to
  FP16, unlike native FP32 execution. This precision change needs new policy/Dev
  validation; it cannot inherit earlier success results by numerical assumption.
- `rotated_channel.py`: adjacent block rotation/channel preparation and output
  cast/rotation/bias fusion. Native intermediate rounding and bias order remain
  explicit; short rows use native CUDA BMM. Per-channel scales are not changed to
  per-token scales. This reduces dispatch/temporary costs, not GEMM precision.
  Callers still own any preceding smoothing/permutation and policy validation.
- `../block_rotation.py`: fused uniform block products with optional permutation,
  without constructing a dense transform. Callers own validated matrices/indices.
  At most sixteen rows explicitly retain native CUDA BMM reduction order.
  Both primitives cache compiled dispatch per instance, not tensor values. Dynamic
  token length must not cause recompilation. No method changes engine implicitly;
  whole-policy and Dev/timing checks are required before a new runtime is admitted.

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
