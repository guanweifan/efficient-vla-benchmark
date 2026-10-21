# HoloQ (formerly Omega-QVLA) — official-code variant

Reference: [author source](https://github.com/UCMP13753/Omega-QVLA/tree/3727e2203568db43fc5fba06ee8686c1b47c044f),
[paper v3](https://arxiv.org/abs/2605.28803v3).
This CogACT binding follows the released composite SVD-Hadamard/zigzag,
block-local GPTQ and per-step activation-scale mechanism.

## Mechanism and adaptation

Quantize 224 Llama and 48 DiT attention/MLP projections. Language weights have
independent scales for every GPTQ input block; DiT uses RTN. Calibration uses
balanced RLDS training observations, uniform token sampling and a complete
scale table for the actual reversed native denoising timesteps.

The official-code variant uses channelwise LLM activation scales and DiT output
rotations. The v3 paper instead describes tokenwise LLM A4 and does not separately
specify those output rotations. FP32 reduced SVD retains the left singular basis,
but finite-precision calibration need not reproduce the author's NumPy codes.
This is not paper-v3 execution or bitwise GR00T/pi0.5 parity.

## Selected runtime

The final `holoq.yaml` uses rotation block 16, GPTQ block 128, `runtime: tinychat_fp16`
and `dit_graphs: true`. All 272 targets use packed W4 with native-rounded channel A4,
FP16 kernel scales/products/accumulation and native output dtypes. This is
tile-dequantized floating GEMM, not integer MMA or native-FP32 accumulation.

Input rotation, zigzag gathering and activation preparation are fused; output
cast/rotation/bias preserves native intermediate rounding. No full decoded weight,
input or prediction is cached. Per-step scale views alias the complete table and
are invalidated on state/device changes.

Each of the 48 DiT targets uses one fixed-shape CUDA graph with the
[current-stream TinyChat runtime](../../../model_ops/quantization/README.md#optional-fixed-shape-dit-graphs),
supplied through `--runtime-path`. Every call copies fresh input and the current
timestep's scale, then clones output. Capture is cold-start overhead; replay copies,
transforms and dispatch remain inside timing. Graphs clear on reload/device moves/close
and reject unexpected geometry. Language execution, DDIM10 and H8 are unchanged.

## Calibration and validation

Use `evb calibrate` with `configs/experiments/cogact/holoq.yaml` and RLDS data;
evaluation requires `--artifact-path` and `--runtime-path`. Conversion is irreversible,
so start from a fresh native backend. Legacy handover artifacts are rejected.

Schema-1 artifacts are reusable only for identical calibration settings; calibration
executes native policies independently of the selected packed engine. GPTQ blocks
smaller than 128 are unsupported by the FP16 packing layout. Serialize the complete
scale table, never just the current-step view.

Whole-policy reference, reset and fresh-reload tests are required: teacher-forced
activation-code agreement alone does not establish action parity. Native-precision
diagnostic engines are not the final report runtime. See
[results](../../../../../docs/benchmark_results.md) for Dev, Final and paired timing.
