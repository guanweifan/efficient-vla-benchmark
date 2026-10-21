# DA-MPA — DA-PTQ's independent drift-allocation core

[DA-PTQ](https://arxiv.org/abs/2604.11572v1), section 3.4 and the separate
DA-MPA ablation in Table 4, motivate this scoped CogACT binding.
It is **not full DA-PTQ**: CSRC's weighted covariance objective and associated
SVD preconditioning are excluded because their weighting/regularization recipe
remains underspecified; no whitening/coloring proxy is substituted.

The retained contribution is the seven-coordinate virtual-chain Jacobian,
damped pseudoinverse, translation/rotation-weighted drift sensitivity, normalized
axis-weighted diffusion epsilon loss, gradient ranking and mixed precision.
A training-only 512-observation calibration supplies real normalized H16
demonstration chunks and native cognition; 16 deterministic probes cover all suites.
No optimizer runs and original checkpoint parameters are not updated.
The Jacobian uses raw dataset commands while epsilon targets use the checkpoint's
training normalization, including the absolute open-gripper conversion.

The final preset retains the top nine of ten eligible DiT MLP blocks plus the final two in native
precision. Ranking sums each block's two mean-absolute weight-gradient scores;
ties use block index. The native CogACT DiT is FP32, so preserved blocks remain
FP32, not the paper's BF16. All 224 Llama projections and remaining DiT MLP
projections use real packed W4 and per-input-channel A8. Attention and all other
modules remain native. No OFT analogue substitutes its L1 loss for epsilon loss.

Calibration uses reproducible balanced RLDS training episodes, not Bridge V2
workspace bins; this data adaptation is explicit. Activation percentiles,
per-output RTN and native-dtype arithmetic are disclosed engineering conventions.
The shared sampler supplies complete unpadded training chunks. Allocation ratios
reuse frozen gradient scores.

The runtime is tile-dequantized floating GEMM, not INT8 MMA. All conversions are
timed; there is no floating reconstructed weight cache. Report the scoped component
and exclusions in every table, never infer the full paper fails from this ablation.

The final `da_mpa.yaml` selects TinyChat packed W4 execution after fused
native-rounded channel A8 preparation. FP16 scales/products/accumulation are
explicit; outputs and retained blocks use native dtype. This is not integer MMA
or bitwise FP32 execution. Weight codes, activation grids and gradient ranking
are unchanged. The schema-1 calibration artifact remains independently validated.

Retention 0.9 leaves the least-sensitive eligible DiT block quantized: 226 packed
linears in total. Ranked allocation remains active, not language-only quantization.
At least one eligible block must remain quantized after rounding; all-native
allocations are rejected. The central ledger retains the selection rationale and
measured evidence.
