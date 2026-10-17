# DyQ-VLA — temporal core on native H8

Reference: [paper v2](https://arxiv.org/abs/2603.07904v2).
No implementation of the selected v2 controller has been verified. The
[October 2026 v3 release](https://arxiv.org/abs/2603.07904v3) links public code,
but changes the controller and falls after the agreed literature cutoff; it is
not silently substituted for this scope. The port implements
fixed W4 language projections, dynamic A2/A4/A8/native activation precision,
motion-fineness/angular-jerk fusion, independently calibrated error-bound
thresholds, and the exact stateful Algorithm 1 counter. A rising pending
candidate resets the downgrade counter; only the fused sensitivity is clamped.

All 224 decoder projections are covered. Vision/action heads remain native.
Weights use FP32 per-output RTN calibration with native-dtype reconstruction.
Activation scales are per-input-channel maxima of calibrated frame percentiles.
These quantizer-axis details are engineering choices not specified by the paper.
W4 remains packed; low-bit activations occupy INT8 storage. The kernel performs
tile dequantization and floating GEMM. It does NOT implement the unavailable
author CUTLASS integer-MMA/zero-copy asynchronous engine; low-bit routing alone
does not imply lower latency on this backend. All dispatch/quantization work is timed.

Dispatch uses actually executed environment-native xyz/rotation commands, not
future predictions, and changes precision at H8 chunk boundaries as permitted
by the paper. K therefore counts chunk decisions; the motion/jerk windows count
control actions. Cold start uses A16; history resets per episode/instruction.
Historical percentiles use all previous commands in the current episode.
The paper does not fix cold-start normalization; both offline and online paths
use this identical causal convention.

Calibration takes a checksummed independent native trajectory population via
the dataset CLI argument, NOT the Dev/Final dataset. Every task is attempted
once; failed episodes are retained and never replaced. Uniformly sampled native
query frames from successful full trajectories provide native and quantized
paired commands with identical RNG seeds. Error is mean L2 over xyz/rotation
of the H8 executed prefix; gripper remains excluded and is evaluated closed-loop.
The estimator uses the first conditional-mean error-bound crossing in equal-count
sensitivity bins. This fitting rule is disclosed because the paper leaves its
estimator unspecified; it is not the handover's unsafe maximum-accepted-frame rule.
A negative boundary explicitly disables an unsupported lower-bit interval.
Calibration is frozen before Dev selection; no policy parameter is trained.

One canonical implementation serves both bindings. The independent artifact
schema rejects the old isolated-frame proxy and fake runtime. Fresh native
models are required to undo irreversible quantization. Runtime registration
does not imply completed calibration, Dev-200, timing, or owner-approved Final.

## Explicit mixed-integer runtime candidate

`dyq_vla_mixed_integer` keeps the v2 temporal dispatcher, target modules, H8 and
independent population, but explicitly selects dynamic **per-token symmetric**
activation quantization. A2 uses [-1, 1], A4 [-7, 7], A8 [-127, 127], nearest-even
rounding and an FP32 amax floor of 1e-5. Per-output W4 scales are BF16-rounded;
the A16 branch retains native BF16 activations. These quantizer details remain
disclosed engineering choices, not author-v2-code parity.

One immutable W4 bank serves all four branches: packed A2 is expanded to INT4 by
a separate timed kernel, A4 uses INT4 MMA directly, A8 uses INT8 MMA with register
weight expansion, and A16 uses tile-dequantized BF16 GEMM. No full decoded weight
bank is cached. The author's register-fused A2 and asynchronous zero-copy pipeline
are not implemented. The separately built [shared runtime](../../../model_ops/quantization/README.md#optional-mixed-precision-runtime-build)
is supplied via `--runtime-path`, checked and recorded; no private path is embedded.

The tokenwise quantizer and changed arithmetic require **new schema-3 calibration**
on the same frozen independent native population. Old channelwise schema-2
thresholds, Dev success and latency cannot be reused. Calibration records the exact
binary/source identity; runtime substitution is rejected. No policy training or
additional trajectory collection is implied. Registration alone is not admission.

`dyq_vla_mixed_integer_k1_fp075` is a speedward routing candidate: K=1
removes the nontrivial downgrade delay while preserving Algorithm 1, and the
A16 threshold is 0.75. Motion/jerk sensitivity, cold-start A16 and the independently
fitted A2/A4 boundaries remain active. It is not a fixed-A8 replacement or a
paper-default configuration. All suites use the same settings.

Routing-only candidates may refit thresholds from the identical schema-3
fixed-precision error records when their full calibration dependency audit proves
those measurements unchanged. Preserve the original calibration source, sample
and weight checksums; record the derivation separately in a new artifact. Never
reuse schema-2 errors or fit on Dev outcomes. Each changed routing configuration
still requires its own controller/reference checks, Dev-200 and full-cost timing.
