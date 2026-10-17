# HoloQ (formerly Omega-QVLA) — official-code variant

Reference: [author source](https://github.com/UCMP13753/Omega-QVLA/tree/3727e2203568db43fc5fba06ee8686c1b47c044f),
[paper v3](https://arxiv.org/abs/2605.28803v3).
This binding explicitly follows the released composite SVD-Hadamard / zigzag,
block-local GPTQ and per-step activation-scale mechanism on native CogACT.
It is not a claim of bitwise end-to-end parity with GR00T or pi0.5.

All 224 Llama projections and 48 DiT attention/MLP projections are included.
Language weights retain independent scales for EVERY GPTQ input block; the
handover's single output-row scale adaptation is not used. DiT uses RTN.
The default engine retains native BF16 language and FP32 DiT arithmetic;
the explicitly selected FP16 engine below does not. Both retain DDIM10 and H8.
Calibration uses only balanced RLDS training observations, uniform token sampling,
and a complete table for the actual reversed native denoising timesteps.

The selected official-code variant uses channelwise LLM activation scales and
DiT output rotations. The v3 paper instead describes tokenwise LLM A4 and does
not separately specify those output rotations. These differences are explicit
in the method identity and results; do not label this as the paper-v3 variant.
SVD uses FP32 reduced matrices with the same left singular basis; the author's
NumPy intermediates can still change finite-precision calibration codes. We verify
the equations and pinned solver, not identical author packs.

W4 is physically packed and A4 codes occupy INT8 storage. The LLM dequantizes
tiles for native BF16 GEMM; DiT explicitly decodes each weight into a temporary
FP32 workspace before native FP32 cuBLAS GEMM. Both decode cost and temporary
memory are measured. No reconstructed weight survives the call or is reused.
This is not integer INT4 MMA or a hidden dense fallback.
Every transform, activation conversion and step-dispatch synchronization is timed.
The shared Dev/timing gate, not successful calibration, decides result readiness.

The DiT execution choice preserves the native reduction order: tiny tile-GEMM
differences crossed A4 rounding thresholds and amplified through denoising.
Teacher-forced activation-code agreement alone was insufficient admission;
ordinary whole-policy reference, reset and fresh-reload checks are required.
Rotation-block 16 is an explicit speedward candidate against the reference 64;
both retain fixed W4A4 and the complete composite rotation.

Use the portable CogACT preset and explicit model/dataset/artifact CLI paths.
A fresh native backend is mandatory; hooks are instance-local and detachable,
but weight conversion is irreversible. Existing Omega handover artifacts are
rejected and must not be relabeled with the new schema.

`holoq_fp16_kernel` and its block-16 candidate explicitly select packed TinyChat
W4 execution with FP16 scales/products/accumulation, including the native-FP32
DiT targets. Native channel-A4 rounding, per-step scales, zigzag permutations,
composite rotations, biases and output boundary dtypes remain intact. Adjacent
rotation/activation preparation and output cast/rotation/bias are fused; there is
no per-call full FP32 weight decode in this engine. All 272 targets remain active.
This is neither integer MMA nor bitwise equivalence to the native-FP32 engine.

Existing schema-1 artifacts are reusable only for identical calibration settings:
calibration executes native policies, not either quantized engine. GPTQ blocks
smaller than 128 are unsupported by this FP16 packing layout and are rejected.
The changed arithmetic requires fresh whole-policy/reference, reset/reload,
four-episode smoke, Dev-200 and full-cost timing; no prior success is inherited.
Only the complete scale table is serialized. The current timestep's scale view
is transient, so reloading after another denoising step cannot overwrite the table.

The FP16 preparation also gathers the zigzag permutation inside the rotation
kernel, without a separate selected-input tensor. Per-timestep scale views are
cached only as aliases of the complete table; device/dtype moves and state reload
invalidate them, and direct table replacement rebuilds them. No input, output or
decoded weight is cached. This is an arithmetic-preserving dispatch optimization,
not a new quantizer; full-policy/reset/reload checks and fresh timing remain gates.

`holoq_fp16_graph_b16` explicitly enables fixed-shape CUDA graphs for the 48 DiT
quantized projections, using a checksummed current-stream TinyChat derivative
supplied through `--runtime-path` (build instructions in the shared operations
README). Language execution is unchanged. Every call copies fresh inputs AND
the selected timestep's scale vector before replay; one graph per operator does
not freeze the first timestep's quantizer. Outputs are cloned, not reused as
policy predictions. No integer-MMA, sampler, rotation or precision change is made.

Graphs clear on state reload/device moves/close; unexpected geometry is rejected.
Capture is cold-start overhead outside warmed timing; all input/scale/output
copies remain inside it. Calibration artifacts are reusable only after whole-policy
exactness checks, and this preset needs fresh smoke, Dev-200 and matched replay
timing. Registration does not imply admission. Per-step graph banks and variable-
length language graphs are not deployed: they add initialization/memory costs
without a reliable additional steady-state benefit in the feasibility checks.
