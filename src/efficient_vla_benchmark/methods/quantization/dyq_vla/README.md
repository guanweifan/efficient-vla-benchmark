# DyQ-VLA — temporal core on native H8

Reference: [paper v2](https://arxiv.org/abs/2603.07904v2).
No official implementation was available at this audit. The port implements
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
