# QuantVLA — CogACT packed runtime

Reference: [pinned official source](https://github.com/AIoT-MLSys-Lab/QuantVLA/tree/9e435e56aac4e743526557e5d9ba6e4fddd6ccca).
The port preserves CogACT's BF16 VLM/FP32 DiT boundaries, DDIM10 and H8.

## Mechanism

The mechanism combines block-SVD input/output rotations, optional energy-smoothed
zigzag permutation, per-row five-point MSE W4 scaling, frozen per-channel A8 scales,
per-head ATM and post-projection OHB. Quantized targets are 224 language and 24 DiT-MLP
linears; twelve native FP32 attention modules receive ATM/OHB. Vision, projectors,
DiT attention weights and sampler precision are unchanged.

`energy_smoothing` affects the optional permutation's column-energy statistic,
not SmoothQuant rescaling. ATM uses the author's clipped ratio/neutral band;
OHB uses a separate log-clipped ratio. W4 codes span -8..7 and A8 spans -128..127.

Calibration uses the four-suite RLDS training pool. Frozen per-channel percentiles
take maxima across 32 observed calls per module; DiT calls include diffusion steps.
Attention statistics use 128 balanced observations with both native and packed
student inference. Completing the percentile window offline avoids the released
wrapper's first-call fallback; this adaptation is not bitwise application parity.

## Selected runtime

The final `quantvla.yaml` selects block size 64, no permutation,
`runtime: tinychat_fp16` and `dit_graphs: true`. Native-rounded channel A8 preparation
feeds packed TinyChat W4 GEMM with FP16 scales/products/accumulation. This is
floating GEMM over packed weights, not integer W4A8 MMA or native-FP32 accumulation.

Adjacent rotation/preparation and output cast/rotation/bias are fused while
preserving intermediate rounding. Short rows retain native CUDA BMM; dynamic token
lengths do not specialize dispatch. Identity smoothing/permutation is skipped, and
static rotation buffers are converted once. Enabled permutations still precede rotation.

Each of the 24 DiT-MLP projections uses one fixed-shape CUDA graph with the
[current-stream TinyChat runtime](../../../model_ops/quantization/README.md#optional-fixed-shape-dit-graphs).
Supply it through `--runtime-path`. Fresh inputs are copied and outputs cloned on
every call; this is not observation/action caching. Graphs clear on state/device
changes and reject unexpected geometry. Capture is cold-start cost; replay copies,
rotations, ATM/OHB and transfers stay timed.

## Calibration and validation

Use `evb calibrate` with `configs/experiments/cogact/quantvla.yaml`, then supply its
schema-3 artifact through `--artifact-path`. Calibration identity includes the engine:
activation percentiles and packed-student ATM/OHB statistics must match. Legacy
handover schema-2 and tile-engine artifacts are not interchangeable with this preset.
Quantization is irreversible; returning to native requires a fresh backend.

Arithmetic-preserving fusion/graph changes require exact whole-policy and reload
checks before artifact reuse; latency needs fresh full-cost timing. See
[results](../../../../../docs/benchmark_results.md) for selected Dev, Final and timing.
