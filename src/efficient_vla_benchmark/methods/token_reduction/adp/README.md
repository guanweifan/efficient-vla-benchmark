# ADP

[Paper](https://arxiv.org/abs/2509.22093v1),
[released implementation](https://github.com/chen7086/VLA-ADP/tree/d7094b09a4996847772c1fa975f09d863e1b759a).
The reference is pinned in `third_party/methods/adp`; retain its MIT/Apache notices.

## Implemented mechanism

Layer-zero RMSNorm and pretrained text-Q/vision-K projections rank visual tokens
before any LLM layer runs. Raw head-averaged logits, without softmax or RoPE,
follow the released `_compute_qk_importance_generic`. Select per-view top-k tokens
with the reference's rounded quotas. The observed-EEF motion gate switches between
full and pruned vision between model calls, using bounded history and a cap on
consecutive pruned windows. No weights, extra encoder or policy training are added.

Use `configs/experiments/openvla_oft/adp.yaml`. All parameters are in config;
resources are supplied through the shared CLI. Control replay is required for
dynamic gating. `enabled: false` or `keep_ratio: 1` leaves native model execution
unmodified; `dynamic: false` is an explicitly static-pruning ablation.

## Source correspondence and adaptations

- Published code uses **observed EEF positions**, not the paper's reconstructed
  predicted-action trajectory. We follow code, using raw proprio XYZ from common
  observation callbacks. The first window is full; the adjacent preset needs
  three completed windows before pruning. The average variant uses the code's
  0.5 EMA, not the paper's cumulative mean.
- A window includes the observation before its first action and after its last.
  The shared controller observes the latter before the next query, giving the
  same causal gate decision without adding simulator access to the method.
- Only the released layer-zero scoring path is implemented. The reference's
  `qk_layer > 0` runs an additional full model pass; it is not silently enabled.
- Preserve the joint checkpoint's original RoPE positions, bilateral mask and
  one-token-offset action readout. The author's sequence rebuild compacts positions;
  our explicit native-position policy avoids modifying protected model semantics.
- The reference preset uses 0.75 retention and 0.4/0.6 view weights across all suites.
  Dev-selected presets vary retention, not the scorer, view allocation or motion-gate rule.
  Preserve per-view quota clipping; actual retained counts, not nominal ratios,
  are recorded. Clamp tiny total budgets to one token per view to avoid the
  reference allocator's otherwise nonterminating correction loop.
- Keep Q/K scoring and top-k on GPU; no full attention matrix or full-model probe.
  All scoring, gate and indexing costs remain within policy timing.
- Cold start and every episode/instruction reset discard history. No model calls
  are skipped or extra replanning requested. No other acceleration method is used.

Tests compare the scorer and observed-motion gate against extracted pinned source,
exercise invalid configs and bounded state, and verify real-model native/disabled
parity and active pruning. Timing and four-episode smoke are recorded centrally in
`docs/benchmark_results.md`; they are not a claim to reproduce paper success rates.

## CogACT binding

`implementations/cogact/` reuses the same released Q/K scorer and observed-EEF
motion controller; it never imports the OFT decoder. One 256-patch camera uses
weight [1.0]; the reference preset retains 0.75 (192 when pruning is active).
The selected retention ratio is recorded in the primary-selection manifest.
Language-Q includes the native nonvisual INPUT embeddings, including BOS,
prompt/separator and cognition EOS, following the reference's non-action-token
selection. This is not contextualized cognition/action attention. Native causal
visibility, original patch/RoPE positions and the final cognition readout remain
unchanged. No model output or diffusion settings are substituted.

Dynamic ADP requires observed EEF XYZ from the shared observation history.
CogACT itself remains image-and-language-only, but this complete acceleration
method additionally consumes that existing sensor input; report the distinction.
Missing observed state fails, rather than using zero or predicted trajectories.
Windows close after the actual executed prefix of eight steps, not after the
sixteen predicted actions. Initialization and dense refresh costs are timed.

Use `configs/experiments/cogact/adp.yaml`. Disabling `dynamic` would remove the motion gate and is not the final full-ADP configuration.
Source/equation checks establish the core selector and controller, not published
checkpoint scores or invariance of performance to different baselines.
