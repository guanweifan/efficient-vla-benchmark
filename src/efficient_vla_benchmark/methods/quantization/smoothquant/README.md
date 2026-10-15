# SmoothQuant W8A8 decoder control

References: [paper](https://arxiv.org/abs/2211.10438) and
[author Llama smoothing/quantization code](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5).
This is a classic control, not a new VLA-specific method.

The independent implementation preserves SmoothQuant's channelwise migration
`s = activation_max**alpha / weight_max**(1-alpha)`: inverse scales are folded
into the two preceding RMSNorms, and the same scales multiply Q/K/V or gate/up
weight columns. Weight maxima are shared across the respective consumer group.
O/down projections are quantized without inventing a nonlinear scale folding.
All seven decoder projections use per-output-channel W8 and dynamic per-token A8;
attention SDPA, vision, projectors and the native action head stay unchanged.
This matches the author's Llama coverage with `quantize_bmm_input=False`, not
an all-component INT8 VLA or an integer attention-softmax implementation.

Calibration uses 32 distinct balanced RLDS training observations by default,
with native multimodal preprocessing, visibility and cognition/action readout.
No optimizer updates, demonstration relabeling or Dev/Final calibration occurs.
Channel statistics, smoothing powers and weight scales use FP32; folded norms
and scaled weights are rounded to the original layer dtype before W8 conversion.
This explicit numerical implementation is not bitwise parity with the author's
low-precision fake-quant simulator. Integer code/scale arithmetic has an independent
reference; dense fake quantization is never the measured runtime.

The shared `IntegerLinear` executes signed INT8 Tensor Core products and INT32
accumulation, with dynamic token quantization and scale/bias epilogue included.
No full floating weight reconstruction, persistent dense copy or silent fallback
is allowed. The same implementation binds independently to OFT's bidirectional
decoder and CogACT's causal decoder; CogACT DiT stays native FP32/DDIM10/CFG1.5.

Calibration publishes checksummed per-layer artifacts only after every native
target executes. Loading requires exact backend/checkpoint/config/target coverage
and original dtype/bias identity. Conversion is irreversible; reload the native
model to undo it. Reset/close own no cross-episode caches.

Use the usual `evb calibrate` / `evaluate` / `benchmark` commands with
`configs/experiments/<backend>/smoothquant.yaml` and caller-supplied model,
dataset and artifact paths. Registration is not admission: whole-policy parity,
LIBERO smoke, original replay timing and Dev-200 must pass before a primary row.
No formal measurements are claimed by this README.

Validation: 18 mechanism/artifact/lifecycle tests pass. On each backend, all 224
integer linears executed across 12 frozen observations from four suites; complete
actions matched an independent Torch INT8/INT32 GEMM oracle exactly, including
reset/reload and gripper checks. Both common four-episode LIBERO smokes completed
4/4 without runtime errors. These are correctness gates, not Dev-200 accuracy or
end-to-end speedup evidence.
