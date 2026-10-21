# Block-wise Adaptive Caching (BAC)

[Paper](https://arxiv.org/html/2506.13456v1),
[reference code](https://github.com/ky-ji/BAC/tree/82029a6fb0573fd07f4b26088219b0eb5ccc5a67).
This independent CogACT port retains BOTH core contributions:

- ACS maximizes accumulated source-to-reused-feature cosine similarity with a
  dynamic-programming interval partition, independently for each residual branch.
- BUA ranks branches by all-pairs feature L1 discrepancy, selects the largest-error
  branches, and unions downstream FFN refresh steps into their schedules.

Use the executable ACS objective from the author solver: score interval [a,b] by
sum of similarity[a,t]. The paper's adjacent-step notation does not fully specify
this source-to-target objective. Step zero initializes every cache; the final
step always refreshes, matching the released runtime's last-step safeguard.
Counts after BUA/final refresh, not nominal ACS budgets, determine actual savings.
Our descending union obeys Eq. 14 even with one protected branch; we do not retain
the author's incidental guard that bypasses BUA for fewer than two selected blocks.

CogACT has twelve pre-normalized DiT layers with attention and MLP residuals, not
the DP-T decoder's additional cross-attention branch. All 24 existing branches
participate. Reuse affects the branch output, never the current residual input.
Norms, final readout, conditioning, CFG pairing, native DDIM steps/RNG and H8 remain
unchanged. Cache state is per-query and per-policy. Full-update and disabled modes
must match native outputs and RNG; close restores every instance-level wrapper.

Statistics come only from balanced RLDS training observations, before evaluation.
One frozen similarity/error artifact supports all budget candidates. We use one
global schedule across all four suites; the author's task-specific schedules are
not imputed. Record this adaptation and calibration time separately.

```bash
evb calibrate --config configs/experiments/cogact/bac.yaml \
  --model-path "$MODEL_DIR" --dataset-path "$RLDS_DIR" --artifact-dir "$BAC_ARTIFACT"
evb evaluate --config configs/experiments/cogact/bac.yaml \
  --model-path "$MODEL_DIR" --artifact-path "$BAC_ARTIFACT" \
  --libero-root "$LIBERO_ROOT" --output-dir "$SMOKE_DIR"
```

Calibration validates input/checkpoint/sampler identities and checksums; original
checkpoints are read-only. No training, success-label fitting or Dev/final
observations are involved. The original 100-step policy speedup is not a claim about CogACT's
10-step sampler. OFT L1 has no iterative denoiser and is not registered.
