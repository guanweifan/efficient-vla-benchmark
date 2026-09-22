# AWQ on OpenVLA-OFT

Reference: [paper](https://arxiv.org/abs/2306.00978),
[pinned implementation](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61).
The actual source commit is recorded by the `third_party/runtimes/llm_awq` Git submodule.

## Scope

Quantize the seven linear projections in each of 32 language decoder blocks to
real packed W4A16, group size 128. The pinned large-M GEMM kernel hard-codes
group size 128, so unsupported 32/64 runtime groups fail explicitly rather than
producing incorrect inference. Vision encoders, multimodal/proprio
projectors, embeddings, LM output head and the L1 action head remain BF16.
This is decoder-weight quantization, not an all-component four-bit VLA model.

The first calibration recipe uses BF16, 32 training observations balanced across
four LIBERO suites, a 20-point activation-power scaling search and the reference
weight-clipping search. It performs no gradient updates or fine-tuning. Calibration
time and derived artifact size are separate from online query latency.

## Files and interfaces

- `config.py`: portable validated settings.
- `implementations/<backend>/target.py` binds decoder layers, attention semantics
  and offloading; other modules below share the same AWQ algorithm.
- `calibrate.py`, `search.py`: capture actual native multimodal
  embeddings and perform layerwise scaling, clipping and packing.
- `artifact.py`: validate/checksum and load all packed layers, including the modified
  normalization parameters. Incomplete, mismatched or corrupt artifacts are rejected.
- `method.py`: attach one quantized runtime to the owning backend.
- `runtime.py`: an explicit precision wrapper around the unchanged author kernel;
  preserves packed INT4 weights and converts only the selected floating buffers,
  inputs and outputs. Converted-buffer overflow is rejected during loading.
- `vendor.py`: load selected pinned AWQ primitives under a private import namespace;
  do not install its unrelated application stack or replace OFT's Transformers.

The shared RLDS reader selects distinct stratified training episodes and records
sample-content digests. RLDS RGB is already in the policy orientation; it is not
rotated again. Variable-length observations form independent padded batch items,
not one long attention sequence. Padding is excluded from scale statistics, search
loss and clipping samples. The author's groupwise quantizer, clipping routine,
packing layout and actual CUDA GEMM/GEMV kernels are reused.

## Commands

Resource variables below are supplied by the caller, never repository defaults.

```bash
evb calibrate --config configs/experiments/openvla_oft/awq.yaml \
  --model-path "$MODEL_DIR" --dataset-path "$RLDS_DIR" \
  --artifact-dir "$ARTIFACT_DIR"

evb evaluate --config configs/experiments/openvla_oft/awq.yaml \
  --model-path "$MODEL_DIR" --artifact-path "$ARTIFACT_DIR" \
  --libero-root third_party/benchmarks/libero --output-dir "$RUN_DIR"
```

Use the same `--artifact-path` with the shared benchmark command. Original model
and data directories are read-only. A calibration failure leaves its output for
inspection; use a new directory for another attempt. `manifest.json` is published
only after all 32 layers have been completed. Quantization cannot be detached to
pretend to restore a dense baseline: construct a fresh backend instead.

## Validation status

The migrated artifact was previously calibrated on 32 real training observations
across four suites. Migration reuses it with full content/identity validation;
it does not claim a new calibration run. GPU tests exercise real BF16/FP16 packed
GEMM/GEMV and load 224 packed linears twice with matching predictions. Fresh smoke
and timing results are recorded centrally in `docs/migration_validation.md`.

The development measurement reduced peak allocated runtime memory substantially,
but did not improve query latency. Packed weight size does not guarantee speedup
at OFT's multimodal sequence length. Do not replace the real result with a
fake-quantization or synthetic GEMV claim. Formal task success rates and broad
calibration-quality comparisons have not been measured. These four episodes are
functional checks, not a success-rate estimate.

## Explicit kernel precision

`kernel_precision: auto` (the unchanged default) follows the model precision.
`awq.yaml` therefore retains BF16 activations and the author's BF16/FP32-accumulation
kernel on the common BF16 model. The optional `awq_fp16_kernel.yaml` selects the
same author's FP16/FP16-accumulation kernel, converting activations at each packed
Linear boundary and returning outputs in the model dtype. All casts are timed.
This is an explicit mixed-precision execution option, not a claim that removing
a software bug alone produced its speedup. Precision, accumulation and boundary
casts are recorded in method metadata. FP16 has a narrower numeric range and
different accumulation error; finite output/smoke checks are not formal accuracy.

Offline scaling/clipping and artifact identity are separate from this runtime
choice. Existing checksummed BF16-calibrated artifacts remain unchanged and can
be used by either option; changing kernel precision does not claim a new FP16
calibration. The artifact still must match the base checkpoint, model precision,
quantization/search parameters and all component checksums.

For a separate native precision control, `baseline_fp16.yaml` changes the whole
native model to FP16 without AWQ. It is not a replacement for the common BF16
reference. Report these conditions separately, never hide a precision switch
inside an AWQ-versus-native speedup.

The source audit and fresh comparisons of both kernel precisions are recorded in
`docs/performance_validation.md`. BF16 remains slower in that retest; the explicit
FP16-kernel option is faster, without a formal task-accuracy claim.
