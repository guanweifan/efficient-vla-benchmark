# TIES — OpenReview revision

Independent implementation target: [OpenReview jN98pgTpd8](https://openreview.net/pdf?id=jN98pgTpd8),
10-page revision supplied by the owner, PDF creation date 2026-05-26, SHA256
`2d1355c44ca0b6df5be2ad0fae6875e88f2bb630f64a218aa3fa96c7563b0cbe`.
Use sections 3.1/3.3/4 and Appendix A together, not a mixture with arXiv v1.
No author-code parity or published-checkpoint score claim is made.

For the first L native language-decoder layers, sum incoming native attention
over query rows and average heads, then rank visual token identities. Compute
Kendall correlation on each adjacent pair's COMMON top-K identities, respecting
their relative order, and average the L-1 correlations. Freeze q25/q50/q75 using
100 balanced training observations, without gradients or evaluation labels.
Low tau trusts attention; high tau prefers uniform retention. Hard-TIES uses q50;
Soft-TIES uses the explicit q25/q75 piecewise-linear interpolation. Attention
receives floor(y * retained_budget) tokens; uniform sampling fills the remainder.

The revision replaces the old similarity-triggered refresh with fixed H control
steps. Figure 2's tau arrows are reversed relative to section 4.1 and Algorithm 1;
the consistent equations and pseudocode, not those arrows, define this port.
Its diagram places pruning after the second full decoder block. Sum incoming
attention as explicitly defined in section 3.1; do not substitute a last-query
FastV score merely because the text calls the top-attention selector FastV-style.

## Explicit implementation conventions

Uniform means seeded sampling without replacement from the unselected original
visual positions, not extra clustering or a diversity scorer. The paper does not
specify a random seed, score-tie policy or deterministic interval sampler; this
port uses an isolated per-episode generator and stable original-position score
ties. Do not claim these numerical conventions are verified author choices.
An adjacent intersection with fewer than two tokens has undefined Kendall tau;
it must never be assigned an invented similarity value. Degenerate soft
calibration quantiles are rejected rather than divided by an arbitrary epsilon.

The reference native H8 preset uses refresh_steps=24, a chunk-aligned setting near
the paper's H=20. H counts executed controls, not decoder calls or diffusion steps.
Additional refresh intervals are explicit Dev-selected parameters in the result
ledger and selected presets, never a claim of the published H=20 recipe.
Current-frame prefix-probe work, rank calculation and selection must all be
included in replay; a future-depth tau must not be obtained free of charge.
Native masks, RoPE, action slots/cognition readout, sampler, CFG and RNG stay native.
Only an evaluated, explicitly frozen result may be promoted to the primary ledger.

## Native calibration transfer

CogACT keeps the revision's K=30 and L=16; all 100 frozen training observations
have defined Kendall scores. On OFT's two-camera sequence, K=30 is undefined
on 55/100 of those training observations. A read-only full-ranking diagnostic
found that K=192 defines all 100 scores (smallest observed adjacent intersection: 2).
The owner approved this explicitly disclosed ranking-window adaptation. OFT's
preset therefore uses K=192, not a claim of the paper's K=30 recipe; its Kendall
formula, layer range, calibration population and allocation rules are unchanged.
K is the correlation ranking window, NOT the retained-token budget. Never drop
undefined calibration frames, insert a made-up tau or use Dev outcomes to fit
the quantiles. Unexpected undefined online intersections retain dense rows and
are explicitly counted in diagnostics, not silently interpreted as tau=0.
The frozen quantiles were independently recomputed from all 100 training scores;
OFT's scores were additionally checked against the saved native rankings.
Calibration retains its actual development-source metadata: OFT's runtime matches
the committed runtime; CogACT differs only in the native test's OFT K setting,
not in calibration or inference code. No dirty measurement is relabeled clean.

## Use

Select the backend's `configs/experiments/<backend>/ties.yaml`. Calibrate once
from read-only training observations, then reuse that frozen artifact across
retention budgets and hard/soft modes; changing K or L requires new quantiles.

```bash
evb calibrate --config "$PRESET" --model-path "$MODEL_DIR" \
  --dataset-path "$RLDS_DIR" --artifact-dir "$CALIBRATION_DIR"
evb evaluate --config "$PRESET" --model-path "$MODEL_DIR" \
  --artifact-path "$CALIBRATION_DIR" --libero-root "$LIBERO_ROOT" \
  --output-dir "$RUN_DIR"
```
