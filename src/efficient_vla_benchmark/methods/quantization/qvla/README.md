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
locations. The result contains schema-4 manifest, sensitivity scores and gates;
`--artifact-path` validates source/config/checkpoint, checksums and full coverage.
Legacy handover gates used a different allocator/covariance normalization and
are rejected for formal evaluation. Conversion is irreversible and native reload
is mandatory before another method. Saved sensitivities can support new channel
budgets without recollecting observations, with a new immutable artifact identity.
