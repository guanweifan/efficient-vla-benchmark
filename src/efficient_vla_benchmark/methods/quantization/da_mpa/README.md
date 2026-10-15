# DA-MPA — DA-PTQ's independent drift-allocation core

[DA-PTQ](https://arxiv.org/abs/2604.11572v1), section 3.4 and the separate
DA-MPA ablation in Table 4, motivate this scoped CogACT binding.
It is **not full DA-PTQ**: CSRC's weighted covariance objective and associated
SVD preconditioning are excluded. The handover's whitening/coloring heuristic
does not implement Eq. 7, and the paper leaves its weights/regularization recipe
insufficiently specified. That proxy is not imported or silently substituted.

The retained contribution is the seven-coordinate virtual-chain Jacobian,
damped pseudoinverse, translation/rotation-weighted drift sensitivity, normalized
axis-weighted diffusion epsilon loss, gradient ranking and mixed precision.
A training-only 512-observation calibration supplies real normalized H16
demonstration chunks and native cognition; 16 deterministic probes cover all suites.
No optimizer runs and original checkpoint parameters are not updated.
The Jacobian uses raw dataset commands while epsilon targets use the checkpoint's
training normalization, including the absolute open-gripper conversion.

The top 30% of eligible DiT MLP blocks plus the final two blocks retain native
precision. Ranking sums each block's two mean-absolute weight-gradient scores;
ties use block index. The native CogACT DiT is FP32, so preserved blocks remain
FP32, not the paper's BF16. All 224 Llama projections and remaining DiT MLP
projections use real packed W4 and per-input-channel A8. Attention and all other
modules remain native. No OFT analogue substitutes its L1 loss for epsilon loss.

Calibration uses reproducible balanced RLDS training episodes, not Bridge V2
workspace bins; this data adaptation is explicit. Activation percentiles,
per-output RTN and native-dtype arithmetic are disclosed engineering conventions.
The handover's end-padded chunks are replaced with complete unpadded training
chunks from the shared sampler. Allocation ratios reuse frozen gradient scores.
Neither calibration nor this optional port authorizes Final evaluation.

The runtime is tile-dequantized floating GEMM, not INT8 MMA. All conversions are
timed; there is no floating reconstructed weight cache. Report the scoped component
and exclusions in every table, never infer the full paper fails from this ablation.

`da_mpa_fp16_kernel` explicitly selects TinyChat packed W4 execution after fused
native-rounded channel A8 preparation. GEMM scale storage, products and accumulation
use FP16; outputs and biases use the native layer dtype, and retained DiT blocks
remain FP32. This is not integer MMA or bitwise FP32 execution. Weight codes,
activation grids, sensitivity scores and allocation do not change. The schema-1
artifact is reusable because its calibration runs only the native model, never
the selected packed engine. New whole-policy checks, Dev-200 and full-cost timing
are compulsory; no old success or speed result is inherited by this preset.

`da_mpa_fp16_kernel_r10` is the speedward allocation candidate: retain the top
one of ten eligible DiT blocks plus the final two, quantizing 242 linears in total.
It reuses the same frozen gradient ranking, not a new calibration or a uniform
allocation. Lower positive retention ratios select the same one protected block;
removing it would disable the ranked-retention contribution.
The realized allocation must also leave at least one eligible DiT block quantized.
A ratio below one is insufficient: rounding can retain every block. Such settings
are rejected instead of silently becoming language-only quantization.
