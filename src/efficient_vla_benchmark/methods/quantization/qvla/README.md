# QVLA — official-code Hessian proxy, OFT packed runtime

Reference: [pinned official source](https://github.com/AutoLab-SAI-SJTU/QVLA/tree/26cc4821a3be4c003d09d3c7997b38db2a347982).
The supported port is OFT LIBERO joint BF16 H8. Registration does not certify
Dev/final results; the central ledger owns measurement status.

Training-only calibration gathers native vision/language Linear and non-grouped
Conv2d input covariance. The released recurrence normalizes by batch examples,
**not token count**. Dead columns, 1% mean-diagonal damping, inverse-Cholesky
diagonal weighting and per-output-channel candidate quantization errors follow
the public code. This is explicitly the **official-code input-Hessian proxy**,
not a claim to reproduce the paper's action-space intervention formulation.
Covariance accumulation preserves the author's per-batch input scaling before
GEMM, contiguous diagonal reduction for damping, and division by the squared
inverse-Cholesky diagonal. Algebraic rewrites can alter finite-precision scores;
schema-4 summed-Gram artifacts must be recalibrated, never relabeled.

`allocation.py` uses absolute lower-bit sensitivity cost per saved bit, stable
candidate insertion order and a global channel-average budget. It is not
marginal-error demotion or parameter-weighted model bits/weight. All active native
target modules are covered; dormant modules are recorded and retained unchanged.
Projectors, action head, LM head, normalizations and biases retain native precision.

Runtime groups output channels by 0/2/4/8/16 bits and restores original order.
Zero-bit rows retain bias; 16-bit rows retain native floating weights. Quantized
rows use physically packed kernels with tile-local floating dequantization, not
integer MMA or a persistent full floating-weight reconstruction. Conv2d unfolding,
row scatter and conversion are included in timing; compression need not accelerate.
Runtime scale estimation, division and rounding use the checkpoint's native dtype,
matching the official injector; sensitivity estimation remains FP32 as in its
separate calibration script. The previous FP32 runtime rounding is superseded,
not numerically equivalent, and requires fresh Dev/timing evidence.

Explicit `quant_only` diagnostic presets exclude zero-bit pruning and 2-bit rows
while retaining the same frozen sensitivity and channel-wise allocation rule.
They isolate whether aggressive pruning drives collapse; they are NOT evidence
for the full joint pruning/quantization setting. Candidate restrictions and the
channel-weighted budget remain in the artifact/config identity. Reallocation may
only select bits already present in the frozen sensitivity measurements.
Uniform nonzero groups bypass redundant output allocation/scatter. After complete
validation and CPU preparation, conversion offloads each original target before
installing its packed replacement, avoiding simultaneous residency of two models.

Use `evb calibrate` with `openvla_oft/qvla.yaml`, explicit model/dataset/artifact
locations. The result contains schema-5 manifest, sensitivity scores and gates;
`--artifact-path` validates source/config/checkpoint, checksums and full coverage.
Legacy handover gates used a different allocator/covariance normalization and
are rejected for formal evaluation. Conversion is irreversible and native reload
is mandatory before another method. Saved sensitivities can support new channel
budgets without recollecting observations, with a new immutable artifact identity.

## Closed experimental scope

Exact pinned-author allocator parity was checked on all 430 modules, while six
representative covariance matrices and all 30 associated bit-score vectors matched
the author's calibration arithmetic exactly. Complete schema-5 recalibration
still assigns zero bits to nearly every channel of early output/down projections
at the released budget. This is not resolved by the arithmetic repair.

The released 32-observation input-Hessian proxy differs from the paper's
action-Jacobian-guided selective interventions; absolute-cost allocation differs
from marginal error per saved bit, and staged versus immediate demotion descriptions
also disagree. The stated pruning regularizer lacks an executable definition.
After the last bounded source/core review, the full paper is **abandoned for this
experimental batch** (`excluded` in selection), not an admitted negative result
or an outstanding promised implementation. This applies to a prospective CogACT
binding as well; changing the backend does not resolve the estimator definition.
Quantization-only presets remain diagnostics, not certified alternative
reproductions. Arbitrary per-layer pruning caps would change the mechanism.
This closure concerns unverifiable full-core fidelity, not a claim of GPU
incompatibility or proof that the published algorithm fails.
