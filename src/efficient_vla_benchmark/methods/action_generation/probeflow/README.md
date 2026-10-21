# ProbeFlow: CogACT DDIM-coordinate port

[Paper](https://arxiv.org/html/2603.17850v1), equations 5-11 and Algorithm 1.
This independent implementation retains the initial/lookahead velocity probe,
cosine-based step scheduler, two-evaluation sparse path and initial-evaluation
reuse in the dense path. Dense N-step integration costs **N+1** network calls,
not N: the discarded lookahead is explicitly included in diagnostics and timing.
The initial domain-level parameters follow the paper: probe 0.5, epsilon 0.008,
Nmin=2, Nmax=10, increment 2. They are candidates, not selected report results.

CogACT is an epsilon-prediction diffusion model, NOT an FM-trained model. Its
deterministic DDIM admits the coordinate change `r=sigma/(alpha+sigma)`,
`y=z/(alpha+sigma)`. A native step is algebraically
`y_next=y+(r_next-r)*(epsilon_hat-x0_hat)`. In noise-to-data time
`u=1-r/r_start`, the velocity becomes `r_start*(x0_hat-epsilon_hat)`.
We preserve the actual finite-noise start and its scale, CFG and native action
postprocessing. At u=1, y is the clean normalized action. We do not substitute
unit-Gaussian noise at an artificial alpha=0 endpoint or divide by zero.

Adaptive uniform-u evaluations interpolate the original training timestep table
linearly in r. Native Fourier timestep embeddings accept fractional indices;
out-of-training-range vector-field requests are rejected, not clamped silently.
Native-grid equivalence is tested independently of whether the NEW adaptive grid
preserves success. The conversion does not endow the model with FM straight-path
training: any curvature/accuracy/latency finding belongs to this disclosed port,
not a reproduction of the published Evo-1 model's numeric results.

The baseline remains native DDIM10/H8. Discarded native eta-zero random draws are
retained to keep later query noises identical across solver budgets. All probe,
conversion, RNG and synchronization overhead is inside common full-cost timing.
No weights, VLM calls, action dimensions or executed horizons change. OFT's L1
head has no iterative vector field and is not registered.

```bash
evb evaluate --config configs/experiments/cogact/probeflow.yaml \
  --model-path "$MODEL_DIR" --libero-root "$LIBERO_ROOT" --output-dir "$RUN_DIR"
```
