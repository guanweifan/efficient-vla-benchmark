# VLA-InfoEntropy

Independent implementation of [arXiv:2604.05323v1](https://arxiv.org/html/2604.05323v1).
Primary code category: caching; survey cross-reference: 5.1. This method does NOT
delete the protected context or simply replace a pruning score. Low-information
static visual tokens reuse KV; important visual tokens recompute. All visual keys
remain available. No source-code parity or original-paper score claim.

Compute per-patch grayscale histogram entropy (256 bins, explicit rounded BT.601)
and Eq. 4-6 text-distribution entropy from the PREVIOUS query's mean attention over
every head/layer. Lower text entropy gives higher information. The second softmax
in Eq. 4 is retained exactly, not replaced by raw logits or temperature tuning.
It yields very small information values; entropy uses FP64 to avoid cancellation.
All extraction/statistic costs are measured, including full-layer cache statistics.

Eq. 7 moves per-view quotas from (60 visual, 40 semantic) to (40, 60) over 100
EXECUTED control steps; clamp thereafter. T is a fixed config, never the future
episode completion length. Take the union without backfilling overlaps. Of up to
150 most similar static patches per view (threshold 0.996), protect this union.
Reuse the remaining patches with the full VLA-Cache-style layer entropy schedule
(layers 2/6/9/11, weighted upward growth 0.55). These cache-engine defaults are
explicit inherited settings, not specified anew by InfoEntropy. The shared OFT
decoder/IndexedKVStore owns execution; no import of another method implementation.

Interpret the paper's contradictory 'low similarity static' prose as HIGH
inter-frame cosine similarity, consistent with its referenced VLA-Cache mechanism.
Cold, instruction and sequence-layout changes invalidate cache. Per-view allocation,
tie-breaking and grayscale conversion are documented OFT conventions. Full-retention
context is not equivalent to full compute: report skipped QUERY rows separately
from 512 retained visual KV tokens. `enabled: false` or static budget zero is native.

Use `configs/experiments/openvla_oft/vla_infoentropy.yaml`. Compare against native
SDPA for practical speed, optionally against VLA-Cache to isolate the new selector.
Do not hide attention-backend changes, histogram overhead or negative speed results.

The 2026-09-21 re-audit adds an independent NumPy reference for Eq. 4-6 and
exhaustive integer-step checks of Eq. 7, including floor and clamping behavior.
These confirm the explicit second softmax, text-axis entropy and executed-step
allocation; no evidence justified dropping full-layer statistics to improve speed.
The inherited cache-engine schedule and two-camera extension remain disclosed
adaptations, not verified author defaults or a claim of published-score reproduction.
