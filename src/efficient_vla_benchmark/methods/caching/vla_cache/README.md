# VLA-Cache

Reference: [paper](https://arxiv.org/abs/2502.02175),
[author implementation](https://github.com/siyuhsu/vla-cache/tree/a4909880573868dee2769343d52e793c0341678b),
[author Transformers fork](https://github.com/siyuhsu/transformers/tree/9a90a37acacf453433168db8d7769b7ea3c40c06).
Both source repositories are pinned under `third_party` for inspection; the second
Transformers package is not installed into the runtime environment.

## Complete mechanism

The numeric values below describe reference defaults. The final presets use a
static-candidate cap of 256 per view, protecting 50 patches on OFT and 75 on
CogACT. The threshold, entropy equation and reuse-layer schedule are unchanged.

1. Compare current and previous inference-frame RGB patches, select up to 150
   static patches per camera at cosine threshold 0.996.
2. Exclude 100 task-relevant patches per camera using the previous query's layer-15
   text-to-vision attention. Keep the reference's original-index candidate order.
3. Compute normalized attention entropy for every layer, including every active
   query and all keys. Use the previous query's entropy schedule with weighted
   upward growth 0.55 to choose reuse counts at layers 2, 6, 9 and 11.
4. Skip selected visual query rows and their MLP computation. Update the active KV
   positions in place while retaining cached rows; attention still sees all keys.

The first query is dense. Episode, instruction and sequence-layout changes
invalidate state. Nothing is appended across frames. Empty candidate sets are a
valid dense-refresh case, not an error. Nonvisual tokens always update.

`config.py` validates settings; shared `method.py` owns the complete policy and
cache lifetime. `implementations/<backend>/` supplies native text/image boundaries;
backend adapters own indexed KV, positions and action readout. No other method is imported.

## Adaptation and measurement

- Use the common joint-suite checkpoint and evaluation protocol; do not claim the
  original paper's exact success rates or speedups.
- Compare against the actual previous inference frame whose KV is stored. The
  reference rollout assigns `replay_images[-1]` after appending the current frame,
  accidentally comparing the current image to itself; that bug is not inherited.
- Derive text positions from the true prompt/proprio layout instead of a fixed
  34-token range. Keep the native baseline action readout rather than adopting
  the fork's shifted action slice.
- Preserve native bidirectional masking and action boundaries. The default
  `attention_backend: eager` follows the author's attention execution pattern:
  compute probabilities once and reuse them for output and complete statistics.
  `attention_backend: sdpa` remains an explicit comparison path that computes
  separate statistics while retaining fused SDPA outputs. Eager and SDPA have
  different low-precision rounding; both paths are identified in metadata.
  Neither removes layers/query rows from entropy collection, freezes schedules
  or changes task protection. All online overhead remains timed.
- BF16 entropy arithmetic follows the reference. FP16 runs upcast this statistic
  to FP32 to avoid underflow of its small epsilon; this is a numeric adaptation.
- Remove reference visualization/printing/internal timers, not mechanism costs.

Use `configs/experiments/openvla_oft/vla_cache.yaml` with the common LIBERO and timing
commands. Default control replay preserves action-chunk execution. Query/prepared
diagnostics are possible but must not be mixed with control-replay comparisons.

Tests compare patch ordering and entropy schedules to the pinned author helpers,
and check cold dense parity, exact identical-frame reuse on a small decoder,
fixed-size caches and instruction/episode invalidation. GPU, LIBERO and timing
checks must also pass before claiming a completed integration. A speedup is not
assumed: attention-statistics overhead can outweigh savings on a particular GPU.

Real-model tests check eager cold/reset parity against an independent
LlamaAttention reference at 1e-6; SDPA uses its native-parity check. Eager must not
call SDPA again. See [results](../../../../../docs/benchmark_results.md) for measured
success and speed against native SDPA, including negative outcomes.

## CogACT binding

The CogACT preset uses the same complete policy, thresholds and entropy equations
with one native 256-patch view. RGB comparison follows the native spatial image
transform and the previous model-query frame. Prompt positions are discovered
from the native visual boundary through the pre-cognition suffix; the final
cognition EOS is never skipped. No action placeholders or OFT action slices are
introduced. Original causal query/key visibility is retained even when the
computed query set becomes sparse and the full KV set remains in storage.

The indexed adapter suppresses the unused append-cache return of CogACT's single
cognition-generation step. Its native generation/readout, FP32 DiT, DDIM10/CFG1.5
and 16-predict/8-execute protocol are unchanged. Full entropy and task-attention
costs are timed. The final preset selects eager, matching the author execution pattern.
The optional SDPA output path retains the same complete statistics.
Compare speed against native SDPA, never solely against a slower eager reference.
A cold eager reference is used only for independent mechanism/parity validation.

`visual_query_tokens_final` counts computed visual queries; `visual_kv_tokens`
counts retained context. These are not token-removal counts. Empty reuse sets and
negative gains remain valid outcomes rather than reasons to omit mechanisms.
