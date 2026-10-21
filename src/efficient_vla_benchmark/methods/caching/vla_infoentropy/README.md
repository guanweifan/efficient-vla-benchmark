# VLA-InfoEntropy

Independent implementation of [arXiv:2604.05323v1](https://arxiv.org/html/2604.05323v1).
Low-information
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

## Runtime

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
Previous eager-version measurements do not describe this SDPA implementation.

## Selected configurations

Use `configs/experiments/openvla_oft/vla_infoentropy.yaml` or
`configs/experiments/cogact/vla_infoentropy.yaml`. OFT uses a static
candidate budget of 192 per view and the reference 40/60 protection quotas;
CogACT uses 150 candidates and 56/84 quotas. See the
[results](../../../../../docs/benchmark_results.md) for selection evidence, Final
success and full-cost timing. Dev eligibility is not a guaranteed final-accuracy
bound; full entropy costs remain included even when there is no speedup.
