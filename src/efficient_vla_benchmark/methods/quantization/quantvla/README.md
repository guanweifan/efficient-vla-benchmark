# QuantVLA — CogACT packed runtime

Identity smoothing/permutation are skipped, and static rotation buffers are cast
once to each native layer's dtype. GPU tests require exact equivalence to the
unoptimized path; calibration, precision and quantization equations are unchanged.

Reference: [pinned official source](https://github.com/AIoT-MLSys-Lab/QuantVLA/tree/9e435e56aac4e743526557e5d9ba6e4fddd6ccca).
The supported port is CogACT LIBERO joint, native BF16 VLM, FP32 DiT and DDIM10 H8.
Registration is not evidence of completed Dev/final evaluation; see the result ledger.

The complete selected mechanism is block-SVD input/output rotations, optional
energy-smoothed zigzag permutation, per-row five-point MSE W4 scaling, frozen
per-input-channel A8 percentile scales, per-head ATM and post-projection OHB.
There are 224 language and 24 DiT-MLP quantized linears and 12 native FP32 attention
modules. Vision, projectors, DiT attention weights and sampler precision are unchanged.

`energy_smoothing` smooths the column-energy statistic used for the optional
permutation. It is **not** SmoothQuant activation/weight rescaling. ATM uses the
author's clipped ratio and neutral band; OHB uses a separate log-clipped ratio.
W4 codes are -8..7; A8 codes are -128..127. Scales use the native module dtype.

Calibration uses only the fixed four-suite RLDS training pool. Per-channel
activation percentiles are maximized across 32 observed calls per module, then
frozen. DiT calls include diffusion steps, not 32 distinct episodes. Attention
statistics use 128 balanced training observations for both native and packed
student inference. The released wrapper can freeze its first-call fallback;
this port explicitly completes its intended percentile-calibrator window offline.
That engineering adaptation is disclosed, not called bitwise application parity.

Runtime weights are physically packed W4 and activations are INT8. The Triton
kernel locally dequantizes tiles for floating GEMM with FP32 accumulation:
**not integer W4A8 MMA**, not a full floating-weight cache, and not guaranteed faster.
Rotations, A8 conversion, ATM/OHB and transfers remain inside common timing.

Use `evb calibrate` with the portable `cogact/quantvla.yaml` preset, explicit model,
dataset and artifact destinations. Evaluation requires the resulting schema-3
artifact through `--artifact-path`. Changing any calibration parameter requires
a matching artifact. Legacy handover schema-2 artifacts are diagnostic-only and
rejected: their scale/permutation/attention formulas differed from the author code.
Conversion is irreversible; returning to native requires a fresh backend.

The explicit `quantvla_fp16_kernel` preset selects the same core with a different
execution engine: native-rounded channel A8 followed by packed TinyChat W4 GEMM
with FP16 scales/products/accumulation, and fused block rotations. Native BF16 LLM
and FP32 DiT boundaries, biases, attention, ATM/OHB and sampler remain unchanged.
This is not integer MMA or bitwise native-FP32 execution. The calibration identity
includes this engine: activation percentiles and packed-student ATM/OHB statistics
must be regenerated; an old tile-engine artifact is rejected rather than relabeled.
Registration alone does not admit its performance or inherit earlier Dev results.

The FP16 engine fuses input rotation/channel preparation and output cast/rotation/
bias into adjacent kernels, preserving native intermediate rounding. Smoothing
and permutations, when enabled, are still applied before rotation. Short row
counts retain native CUDA BMM; dynamic token lengths do not specialize dispatch.
The tile engine and calibration equations remain unchanged. Engine artifacts
can be reused for this arithmetic-equivalent optimization only after exact
whole-policy/reference and reload checks; latency requires fresh full-cost timing.

`quantvla_fp16_graph` explicitly enables one fixed-shape CUDA graph for each of
the 24 DiT-MLP quantized projections. Its stream-correct TinyChat derivative is
checksummed and supplied via `--runtime-path`; see the shared operations README
for the portable build. Language, native attention, ATM/OHB, quantizer, sampler
and precision remain unchanged. Fresh inputs are copied on every call and outputs
cloned before scratch reuse; no robot observation/action cache is introduced.

Graphs clear on state reload/device moves/close and reject unexpected geometry.
Capture is separately disclosed cold-start overhead, not warmed inference; every
replay copy stays timed. Same-arithmetic artifact reuse requires exact full-policy
and reload checks. The graph preset receives fresh smoke, Dev-200 and original
replay timing before admission; no accuracy or latency is inherited automatically.
