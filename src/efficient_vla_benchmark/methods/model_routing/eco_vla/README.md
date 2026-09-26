# EcoVLA

Independent implementation of [EcoVLA](https://arxiv.org/html/2602.00780v1),
Sections 3-4. No official executable implementation was available at integration;
this is a paper-core reproduction, not author-code parity or published-score parity.

## Mechanism

1. Compare consecutive native visual-encoder token features with mean tokenwise
   cosine similarity (Eq. 5). Trigger a dense update when similarity falls below
   the configured quantile of PREVIOUS similarities (Eqs. 6-7).
2. At a triggered query, run the native dense policy. Buffer the inputs of every
   attention output projection and MLP down projection on a separate CUDA stream.
3. Compute all-channel squared activation energies (Eq. 8), fuse historical and
   current energies using alpha (Eq. 9), and update history with a DISTINCT lambda
   (Eq. 10). Eq. 11 uses `E * sqrt(sum(W**4))`, not the Wanda column L2 norm.
4. Keep coupled gate/up rows and down columns, and whole Q/K/V heads with matching
   output columns. Publish new masks only for the NEXT query. Pruning-stream
   batched metrics overlap the native action head; synchronization is inside the
   full-cost timing boundary. No extra policy call or action replacement occurs.
5. Execute real indexed Triton GEMMs, with fused gate/up/SiLU/product and
   column-major output weights. No dense masked GEMM or per-query weight gathering.

This is fine-grained dynamic channel/head pruning, NOT whole-layer skipping.
It is filed with conditional model execution and cross-referenced to model
compression/kernel optimization; the paper is implemented once.

## Explicit port choices

The appendix gives OFT/CogACT `p=5` (interpreted as the fifth percentile) and
`alpha=0.7` for CogACT and two OFT suites. We use one `alpha=0.7` across ALL four
suites. The paper does not give a complete runnable recipe for window size,
lambda, initial history handling or head grouping. Our configurable conventions
are window 16, lambda 0.9, the first two queries dense, then an available-history
quantile until the window fills. Heads use the joint L2 score of their channel
scores; FFN retention is rounded upward to 64-channel blocks. These are disclosed
implementation choices, not claimed author defaults. The selected count can differ
slightly from the requested retention ratio and is recorded.

Historical energies initialize from 32 equally weighted training observations
(eight per suite, seed zero) sampled by the common RLDS reader. This is calibration,
not gradient training. Episode/instruction reset restores the frozen calibration
prior and clears visual history, so no evaluation outcomes train the method.
Artifacts checksum energies, checkpoint, source, sample and precision identities.

Only the native OFT and CogACT MHA Llama decoders are supported initially. Masks,
RoPE positions, final readout, action heads and H8 remain native. No-op configurations
leave the model untouched; active close joins all work, removes instance wrappers
and restores original weight layout. Original checkpoints are never modified.

## Validation and results

Use the common `calibrate`, `evaluate` and compact `benchmark` commands with
`configs/experiments/<backend>/eco_vla.yaml`; pass assets as CLI arguments.
Calibration is shared across retention candidates, but each candidate needs its
own Dev-200 and timing pair. Tests cover prior-only trigger history, separate
temporal coefficients, coupled groups, real sparse kernels and native/reset/RNG
boundaries. Full-cost speedups must include dense update queries and every stream.
Do not infer speed from nominal sparsity or the paper's original suite checkpoints.
