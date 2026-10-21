# DyQ-VLA — v2 temporal core

Independent implementation of [paper v2](https://arxiv.org/abs/2603.07904v2).
No author implementation of this v2 controller was verified. The
[v3 release](https://arxiv.org/abs/2603.07904v3) changes the controller and falls after
the literature cutoff; it is not substituted here.

## Mechanism

All 224 language projections use fixed W4 weights with dynamic A2/A4/A8/native
activation precision. Motion-fineness/angular-jerk fusion and Algorithm 1's stateful
counter select precision from actually executed xyz/rotation commands. A rising
pending candidate resets the downgrade counter; only fused sensitivity is clamped.
Decisions occur at H8 chunk boundaries, while motion windows count control actions.
Cold start uses A16; history resets per episode/instruction. Causal normalization
uses previous commands only, an explicit convention where the paper is underspecified.
Vision and action heads remain native.

## Selected runtime

Both final `dyq_vla.yaml` presets use `runtime: mixed_integer` and dynamic
per-token symmetric activation quantization: A2 [-1, 1], A4 [-7, 7], A8 [-127, 127],
nearest-even rounding and an FP32 amax floor of 1e-5. W4 uses per-output RTN scales
rounded to BF16; A16 retains native BF16 activations. These axes and arithmetic are
disclosed implementation choices, not author-code parity.

One immutable packed W4 bank serves every branch. A2 expands to INT4 in a separate
timed kernel; A4 uses INT4 MMA; A8 expands weights in registers for INT8 MMA;
A16 uses tile-dequantized BF16 GEMM. No decoded weight bank is cached. The author's
register-fused A2 and asynchronous zero-copy pipeline are not implemented.

Build the [mixed runtime](../../../model_ops/quantization/README.md#optional-mixed-precision-runtime-build)
and supply it through `--runtime-path`. All dispatch, conversion and synchronization
costs remain timed. Reverting quantization requires a fresh native model.

## Calibration and use

Calibration requires a checksummed independent native trajectory population,
not RLDS frames or Dev/Final episodes. Attempt each task once, retain failures
without replacement, and sample native query frames from successful trajectories.
Native and quantized commands share RNG seeds. Fit the first conditional-mean
error crossing in equal-count sensitivity bins; error is mean L2 over xyz/rotation
of the executed H8 prefix, excluding gripper. A negative boundary disables an
unsupported low-bit interval. This estimator is an explicit paper-to-code convention.

The mixed runtime requires schema-3 calibration bound to its binary/source,
checkpoint, population and quantizer. Old channelwise schema-2 errors, thresholds
and measurements are incompatible. No policy training occurs.

Use `evb calibrate` with `--dataset-path` pointing to the frozen population;
evaluate with `--artifact-path` and `--runtime-path`. Both final presets use K=1;
the A16 threshold is 0.75 for OFT and 0.65 for CogACT. Independently fitted A2/A4
boundaries remain active; this is not a fixed-A8 replacement. The calibration
action-error budget is distinct from the Dev success-loss target.

Routing-only changes may refit thresholds from identical schema-3 fixed-precision
error records after a dependency audit. Preserve original source/sample/weight
identities, record the derived artifact, and remeasure Dev and full-cost timing.
See [results](../../../../../docs/benchmark_results.md) for the selected outcomes.
