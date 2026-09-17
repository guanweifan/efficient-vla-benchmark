# SpecPrune-VLA on OpenVLA-OFT

Reference: [paper](https://arxiv.org/abs/2509.05614),
[author code](https://github.com/alexwhz-sjtu/SpecPrune-VLA/tree/8091adc4b574ce9008d49a1dc9a210f4eec314c1).
The fixed reference is available under `third_party/methods/specprune_vla`.

## Mechanism and ownership

This directory implements one complete method: previous-step global-token reuse,
frame-similarity/local text-guided pruning in the first two layers, action-attention
rank/entropy/EMA scoring with progressive pruning, and the action-aware controller.
The controller retains six raw control frames, chooses the reference's motion-
dependent lookback, switches profiles and requests early replanning. It never
steps the simulator itself. Optional layer skipping is not enabled in the author
default and is not implemented here.

`config.py` owns validated parameters; `implementations/openvla_oft/method.py` owns state and
decisions. The backend decoder owns token-layout/RoPE/attention/cache primitives;
the common controller owns execution. No other method's code is imported.

Use `configs/experiments/openvla_oft/specprune_vla.yaml` with the shared evaluation and
benchmark commands. Timing requires `--mode control` and complete schema-2 traces;
sparse query-only replay cannot reproduce the lookback/controller mechanism.
Internal switches exist for ablations; all three mechanisms are enabled by default.

## Explicit adaptation differences and upstream quirks

- One joint-suite checkpoint and the common native LIBERO protocol are used;
  results are an adaptation, not a claim to reproduce the paper's exact numbers.
- Preserve the original OFT checkpoint's action-head readout, including its
  one-token offset. The author's fork changes that slice; changing our baseline
  or its action head to follow the fork would confound an acceleration comparison.
- Global-reuse selection follows the pinned code's text-attention helper, rather
  than relabeling it action attention. Layer-wise importance uses action attention.
  Replace its hardcoded 34-token instruction bound with the actual text boundary;
  account explicitly for the proprio token. Prefix skips retain the reference
  offsets; very short text ranges retain at least one text query.
- Keep the reference schedule: pruning at layer 10 precedes the first importance
  update at layer 14. Its first dynamic selection therefore sorts tied scores.
  This is retained and documented, not silently replaced with another algorithm.
- The pinned "precise" profile actually has smaller text budgets and larger
  unchanged-patch budgets. Defaults follow the code; the name alone does not prove
  more conservative retention. All relevant budgets are exposed in configuration.
- Enforce a true minimum of 60 visual tokens in dynamic pruning, rather than the
  reference's prompt-length formula that is off by one for the proprio layout.
- Every emitted action is executed once. Do not inherit the author's rollout's
  missing `env.step` for the first nine controller-enabled steps.
- Use native bidirectional SDPA with selected attention-row probes, rather than
  forcing full eager attention at every layer. Probe/selection/controller costs
  remain inside the shared timing measurement; the dense baseline stays native.

## Verification

Tests cover reference patch-selection equivalence, protected global/nonvisual
tokens, dynamic budgets, per-episode confidence reset, controller transitions and
dense equivalence with internal mechanisms disabled. Real-model tests also verify
all pruning mechanisms run and an episode reset reproduces the same commands.

Development validation on the joint-suite checkpoint completed four real LIBERO
episodes without runtime errors: Spatial, Object and Long succeeded; Goal reached
the step limit. Preserve that failure rather than retrying until success. This
four-episode smoke is not a success-rate estimate.

Fresh migration timing is recorded centrally in `docs/migration_validation.md`.
Report both per-query latency and total replay time: extra controller queries can
reduce the overall benefit. These are not closed-loop task-completion times.
