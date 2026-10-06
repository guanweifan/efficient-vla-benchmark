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
are unchanged. Native SDPA produces the policy attention output. Separate complete
probabilities supply the statistics; both computations remain in full-cost timing.
Cold/reset/disabled/detach checks, full-key versus sparse-query tests and executed
step quota tests accompany this independent paper-based port.


## Engineering revision

The default is native SDPA, removing eager-versus-SDPA output differences on
cold or fully protected frames. Eager remains an explicitly selectable diagnostic.
All layer/head statistics and the second softmax are retained. Compact per-layer
requests return only complete-row entropy and the full prompt-to-visual rectangle,
avoiding repeated CUDA boolean indexing and retaining no unnecessary full map
after the layer. Layer schedule entropy is accumulated in FP32 instead of BF16;
the much smaller semantic information signal still uses FP64.

Static candidate order, original masks/positions, monotonic within-query reuse,
per-view protection unions and bounded indexed KV storage are unchanged.
There is no automatic cross-frame refresh or changed control cadence.
This is an engineering revision of the same paper-based mechanism; previous
eager-version accuracy/timing must not be relabeled as this implementation.

## Selected Dev configurations

Use `configs/experiments/openvla_oft/vla_infoentropy_sdpa192.yaml` or
`configs/experiments/cogact/vla_infoentropy_protect140.yaml`. OFT uses a static
candidate budget of 192 per view and the reference 40/60 protection quotas;
CogACT uses 150 candidates and 56/84 quotas. Frozen Dev-200 gives 195/200 and
177/200, respectively, at the fastest eligible measured settings. CogACT is
exactly at the native-relative 3-percentage-point loss boundary. The 75-candidate
CogACT alternative gives 184/200 with slightly lower measured speed and remains
search evidence, not a second primary configuration.

OFT has no reliable end-to-end acceleration at the accuracy target. Report that
negative result; do not omit entropy costs or transfer old eager-version Final
scores. Current paired timing and evidence identities are in
`docs/benchmark_results.md`; this revision has not undergone Final-2000.
