# HoloQ (formerly Omega-QVLA) — official-code variant

Reference: [author source](https://github.com/UCMP13753/Omega-QVLA/tree/3727e2203568db43fc5fba06ee8686c1b47c044f),
[paper v3](https://arxiv.org/abs/2605.28803v3).
This binding explicitly follows the released composite SVD-Hadamard / zigzag,
block-local GPTQ and per-step activation-scale mechanism on native CogACT.
It is not a claim of bitwise end-to-end parity with GR00T or pi0.5.

All 224 Llama projections and 48 DiT attention/MLP projections are included.
Language weights retain independent scales for EVERY GPTQ input block; the
handover's single output-row scale adaptation is not used. DiT uses RTN.
Native BF16 language and FP32 DiT arithmetic, DDIM10 and H8 remain unchanged.
Calibration uses only balanced RLDS training observations, uniform token sampling,
and a complete table for the actual reversed native denoising timesteps.

The selected official-code variant uses channelwise LLM activation scales and
DiT output rotations. The v3 paper instead describes tokenwise LLM A4 and does
not separately specify those output rotations. These differences are explicit
in the method identity and results; do not label this as the paper-v3 variant.
SVD uses FP32 reduced matrices with the same left singular basis; the author's
NumPy intermediates can still change finite-precision calibration codes. We verify
the equations and pinned solver, not identical author packs.

W4 is physically packed, A4 codes occupy INT8 storage, and the common kernel
dequantizes tiles for floating GEMM/FP32 accumulation. No full floating weight
cache, integer INT4-MMA claim, extra low-rank branch or hidden dense fallback.
Every transform, activation conversion and step-dispatch synchronization is timed.
The shared Dev/timing gate, not successful calibration, decides result readiness.

Use the portable CogACT preset and explicit model/dataset/artifact CLI paths.
A fresh native backend is mandatory; hooks are instance-local and detachable,
but weight conversion is irreversible. Existing Omega handover artifacts are
rejected and must not be relabeled with the new schema.
