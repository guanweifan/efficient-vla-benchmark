# QuantVLA runtime feasibility components

Status: **not registered, not a completed reproduction, not a formal result row**.
Reference: [official implementation](https://github.com/AIoT-MLSys-Lab/QuantVLA/tree/9e435e56aac4e743526557e5d9ba6e4fddd6ccca).

`PackedDuQuantLinear` executes calibrated smoothing, input permutation, block
rotation, packed W4/A8 linear, output rotation and post-rotation bias. Uniform
rotation blocks use batched matrix multiplication; ragged blocks remain explicit.
All transformations are online work and must remain inside timing boundaries.
The shared kernel retains each input channel's activation scale inside reduction.
W4 is physically packed and A8 is physically stored as INT8, but multiplication
uses tile-local dequantized floating operands and FP32 accumulation: **not integer
W4A8 MMA**, and no claim of native INT4/INT8 Tensor Core acceleration.

The CogACT model diagnostic verifies a checksummed 248-linear calibration artifact:
224 language linears and 24 DiT MLP linears. It keeps the artifact's 12 FP32
DiT-attention ATM/OHB transformations. Dense reconstruction exists only in the
private correctness oracle; production primitives retain no full floating weight
cache. The original artifact's fake-runtime label is not rewritten or passed off
as proof of real execution; the derived probe records its actual packed kernel.

Tests establish full-precision transform identity and packed/reference agreement
for BF16 and FP32, including input/output rotations, smoothing, permutation and
bias. Whole-model parity is approximate floating arithmetic, not bitwise equality
to the dense oracle; reset reproducibility is checked separately and exactly.

Before admission, move the validated attention binding and artifact checks into
the method interface, expose typed configuration, verify complete reload/identity,
and run the shared compact replay. The prototype does not register a policy,
expose a selectable preset, or establish formal success/speed. Preserve native
CogACT FP32 DiT and sampler; neither OFT support nor a faster runtime is implied.
