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

## CogACT binding

`configs/experiments/cogact/vla_infoentropy.yaml` uses the identical shared selector,
quota equations and cache schedule on one 256-patch native policy RGB grid. Text
entropy averages every layer/head over prompt rows after the visual prefix and
before cognition EOS. No action-token proxy is used. The native cognition row is
always recomputed; original per-query causal visibility and all visual keys remain.
The existing causal indexed-KV adapter owns execution, independently of VLA-Cache's
method implementation. DDIM, checkpoint weights and native spatial preprocessing
are unchanged. Eager probabilities are shared with the statistics; their precision
and overhead are reported rather than claimed identical to native SDPA.
Cold/reset/disabled/detach checks, full-key versus sparse-query tests and executed
step quota tests accompany this independent paper-based port.

Readiness candidate `cogact/vla_infoentropy_static100` changes only the maximum
static reuse candidate count from 150 to 100. Complete histogram, text entropy,
quota transition, task protection and layer schedule remain present. This is a
selective accuracy-sensitivity test, not an already validated replacement default.
