# AWQ decoder quantization

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

Calibration uses BF16, 32 training observations balanced across
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

## Selected runtime and validation

The final `awq.yaml` on each backend explicitly selects `kernel_precision: float16`.
The model and calibration remain BF16. The unchanged author's FP16 kernel uses
FP16 scale storage/products/accumulation; activations are cast at each packed
Linear boundary and outputs returned to native dtype. All 448 boundary casts over
224 linears are timed. The rest of the native model does not become FP16.

`kernel_precision: auto` remains an implementation default, not the final preset.
Offline scaling/clipping and artifact identity are independent of this runtime
choice: the selected runtime reuses checksummed BF16 calibration, not new FP16
calibration. Artifacts must match checkpoint, model precision, quantization/search
parameters and every component checksum. Converted-buffer overflow is rejected.

Real-model checks verify author packed GEMM/GEMV dispatch, fresh reload and matching
predictions. Actual Dev-200, paired full-cost timing and Final-2000 results are in
the central ledger with original identities. Packed weights alone do not guarantee
a speedup; do not replace a measured slowdown with a synthetic GEMV result or
fake-quantized runtime. Thermal/drift-rejected runs are not speed evidence.

## CogACT binding

The CogACT preset captures native single-camera/language/cognition inputs and
uses causal attention during every calibration forward and scale search. It
quantizes only 224 Llama block linears; vision, projector, embeddings, LM head
and the FP32 diffusion head remain native. DDIM10/CFG1.5 and the 16-predict/8-execute
protocol are unchanged. Recalibrate on the joint-suite training data: OFT
artifacts are rejected, never re-labelled or loaded into CogACT.

Use the corresponding `configs/experiments/cogact/awq.yaml` with the same CLI
resource arguments above. Build the pinned AWQ extension inside the CogACT runtime;
do not assume a wheel from another environment has a matching ABI. Verify its
Torch/CUDA/Python build and real kernels here before reuse. Both final presets use
the explicitly selected FP16 kernel with native model boundaries and all casts
timed. Neither performs gradient updates.
