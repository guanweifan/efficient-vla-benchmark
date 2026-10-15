# GPTQ W4A16 decoder control

Reference: [paper](https://arxiv.org/abs/2210.17323) and
[pinned author solver/driver](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf).
This is a classic control, not a VLA-specific contribution or AWQ composition.

The independent solver retains global input Hessians, dead-column handling,
inverse-Cholesky damping, within-block and cross-block error compensation,
and dynamic symmetric group-128 scales. Activation ordering is disabled.
True sequential calibration follows Q/K/V, O, gate/up, then down groups;
subsequent groups and decoder layers observe quantized predecessors. It does
not replace GPTQ with round-to-nearest or independent block-local Hessians.

Default calibration uses 32 distinct balanced RLDS training observations, 1%
damping and 128-column update blocks. Native multimodal embeddings, attention
masks and positions are captured separately for each observation, without padding
or cross-observation attention. Empty prefill caches are disabled only for offline
layer replay. FP32 Hessian/solver arithmetic disables TF32 locally and restores it.
There are no gradient updates, policy training or Dev/Final calibration samples.
The author-style offline dense reconstruction uses native BF16; it is a calibration
oracle, never the reported runtime. Artifacts retain physically packed W4 codes
and FP32 scales, provenance and every layer checksum.

The runtime packs those SAME codes for the pinned TinyChat GEMM/GEMV from
`third_party/runtimes/llm_awq`. Only its execution kernel is reused: no AWQ scaling,
clipping or search is applied. FP16 scale storage/products/accumulation and BF16
input/output boundary casts are explicit and fully timed. This is not integer MMA,
not a dense weight cache and not bitwise native-BF16 accumulation. Scale conversion
overflow/underflow is rejected. Returning to baseline requires a fresh native model.

Only the 224 decoder projections are quantized. Vision, projectors, RMSNorms,
embeddings and native L1/diffusion action heads stay unchanged. OFT retains
bidirectional attention and H8; CogACT retains causal attention, FP32 DiT,
DDIM10/CFG1.5 and sixteen-predict/eight-execute. Calibration artifacts cannot cross
backends or checkpoints. All artifacts are prepared/validated before mutation.

Use `evb calibrate`, then the common `evaluate` / `benchmark` commands with
`configs/experiments/<backend>/gptq.yaml` and explicit model/dataset/artifact paths.
The solver has been compared directly with the pinned author implementation,
including dynamic-group and cross-update-block cases. On each backend, twelve
frozen observations across four suites match independently prepared direct-author
kernel execution exactly, including reset and fresh reload. This verifies the
packed artifact/runtime connection, not native-policy accuracy. The common
four-episode smoke and original replay timing / Dev-200 are separate gates;
consult the result ledger for admitted measurements. Solver or kernel tests alone
do not establish task accuracy or policy acceleration.
