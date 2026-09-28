# TeaCache

Training-free transfer of the **timestep-embedding variant** in
[TeaCache, Section 3.3](https://arxiv.org/html/2411.19108v1), also used by the
[released HiDream port](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4HiDream-I1/teacache_hidream_i1.py).
CogACT has no adaLN modulation. This variant uses its actual timestep embedder,
not invented modulation or a substituted first-block probe. It is not the default
video variant based on timestep-modulated noisy inputs.

On independent balanced training frames, record relative-L1 changes in native
timestep embeddings and the full transformer-stack residual. Fit a degree-four
polynomial across these paired observations. During sampling, accumulate the
polynomial-rescaled input differences; refresh the stack when the threshold is
reached and otherwise add its last residual to the CURRENT stack input. Native
embeddings and final projection remain fresh. Follow the released strict threshold
and initial warmup, without inventing a compulsory final refresh for this variant.

The fixed native noise grid makes this timestep-only signal input-independent.
Precomputing its exact polynomial gate from the checksummed calibration is an
algebraic optimization, not replacing the gate with a periodic interval. It does
not claim online adaptation to action or image changes. Both CFG rows retain
their own residual values and share the gate. Cache state never crosses queries.
Native DDIM, conditioning, CFG, random draws and H8 execution are retained.

Calibration binds the checkpoint, schedule, precision, geometry and actual
training samples. No gradients, success labels or Dev/final observations are used.
Fit error and negative fitted increments are recorded; no video coefficients or
unreported polynomial clamping are substituted. Degree fitting is statistical
calibration, not policy training. Full-cost timing includes residual additions and
all remaining per-step work. Disabled/all-fresh paths install no wrappers.

```bash
evb calibrate --config configs/experiments/cogact/teacache.yaml \
  --model-path "$MODEL_DIR" --dataset-path "$RLDS_DIR" --artifact-dir "$ARTIFACT_DIR"
evb evaluate --config configs/experiments/cogact/teacache.yaml \
  --model-path "$MODEL_DIR" --artifact-path "$ARTIFACT_DIR" \
  --libero-root "$LIBERO_ROOT" --output-dir "$SMOKE_DIR"
```

OFT has no iterative denoising and is unsupported. Registration/calibration alone
is not a successful closed-loop experiment or a measured speedup.
