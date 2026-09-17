# VLA-Cache on OpenVLA-OFT

Reference: [paper](https://arxiv.org/abs/2502.02175),
[author implementation](https://github.com/siyuhsu/vla-cache/tree/a4909880573868dee2769343d52e793c0341678b),
[author Transformers fork](https://github.com/siyuhsu/transformers/tree/9a90a37acacf453433168db8d7769b7ea3c40c06).
Both source repositories are pinned under `third_party` for inspection; the second
Transformers package is not installed into the runtime environment.

## Complete mechanism

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

`config.py` validates settings; `implementations/openvla_oft/method.py` owns selection and cache
lifetime. The shared OFT decoder owns indexed KV storage, position bookkeeping and
the unchanged native action-head boundary. No dependency on another algorithm.

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

Initial development checks passed cold-query native parity at 1e-6, bitwise reuse
of skipped KV rows on real observations, and episode/instruction invalidation.
All four small LIBERO smoke episodes completed successfully without runtime errors;
this is not a formal success-rate estimate.

The standard-attention correction has additional real-model cold/reset parity
checks against an independent eager LlamaAttention reference at 1e-6; the SDPA
option retains its native-parity check. Shared backend tests ensure eager does
not call SDPA again. Matching the author execution pattern does not guarantee
a speedup over an optimized native SDPA baseline; token-count savings are not
necessarily proportional to GPU latency savings.

Fresh migration timing is recorded centrally in `docs/migration_validation.md`.
Retain negative results: do not compare only against an artificially slow eager
baseline or omit the full entropy mechanism to manufacture a speedup.
