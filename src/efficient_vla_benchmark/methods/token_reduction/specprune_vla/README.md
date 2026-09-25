# SpecPrune-VLA

## Bounded development diagnostics

Dev-200 compares two explicit variants on each backend. `specprune_vla_static128`
caps low-change deletion at 128 patches per view in both controller modes,
without disabling global/local selection, dynamic pruning or control. The
`specprune_vla_score_first` schedule adds an importance update at layer 9 before
the first layer-10 pruning; other layers are unchanged. This is an intentional
schedule ablation, NOT a correction to the pinned author's executable defaults.
The latter initialize scores to zero and first update at layer 14.

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

`config.py` owns validated parameters; `method.py` owns shared state and decisions.
Backend bindings own native image preparation and query-token definitions.
The backend decoder owns token-layout/RoPE/attention/cache primitives;
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

Timing and historical cohorts are recorded in `docs/benchmark_results.md`.
Report both per-query latency and total replay time: extra controller queries can
reduce the overall benefit. These are not closed-loop task-completion times.

## CogACT binding

The [v3 paper](https://arxiv.org/html/2509.05614v3#A1.SS4) reports CogACT separately
from an optional action-expert cache. This binding ports all three SpecPrune
mechanisms without that cache: global/local static pruning, progressive rank/entropy
importance and the physical emitted-action controller with early replanning.

Global/local probes use text-to-visual attention as in pinned executable code;
the v3 Eq. 2 image-to-text notation conflicts with that code and native causal
ordering. This implementation follows the code, not an unmasked attention forward.
Dynamic saliency uses the native final cognition EOS, which conditions the DiT.
It is an explicit single-condition backend adaptation, not a claim to observe
action-to-patch attention inside CogACT's DiT or reproduce unpublished author code.

Use `configs/experiments/cogact/specprune_vla.yaml`. One 256-patch grid uses the
primary-camera settings only; wrist parameters do not participate. The global
minimum is 30 rather than the two-camera preset's 60, retaining the per-view floor.
Other released control/selection settings stay fixed, including the early tied-score
pruning schedule. Raw-frame lookback uses native RGB preprocessing without an OFT
crop. Original causal visibility, positions, final cognition and DDIM remain intact.
Reset clears frames, confidence, global saliency and controller state. Probe costs
and any extra model queries are included in control-replay timing.
