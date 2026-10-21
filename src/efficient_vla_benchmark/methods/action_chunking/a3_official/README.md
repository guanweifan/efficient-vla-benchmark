# A³ — official-code version

This scope follows the released GR00T implementation at
[INCEPTIONwang/A3, revision 96a85c6](https://github.com/INCEPTIONwang/A3/blob/96a85c698f3848f6e684e36a81caf6e275db2ecc/rlinf/models/embodiment/gr00t/gr00t_action_model.py),
not the clustering/medoid procedure in [paper v4](https://arxiv.org/html/2605.11567v4).
Use the explicit ID `a3_official` and label **A³ official-code version** in results.
Apache-2.0 attribution and modification notice are in the root NOTICE.

## Retained official mechanism

- Generate independent draft candidates after ONE shared visual-language encoding.
  Compute diagonal Mahalanobis confidence of cumulative execution deltas against
  the global candidate mean/variance, with per-position median confidence.
- Select the candidate with the most above-median positions, then highest masked
  confidence sum, then first index. There is no trajectory clustering or medoid.
- In parallel, regenerate each position conditioned on earlier ranks, in both
  descending-confidence order (stable temporal ties) and temporal order. Accepted
  positions from earlier verification blocks remain conditions. Tentative earlier
  ranks within the current block also condition later rows, exactly as released.
- Each branch accepts until its first axis-wise threshold failure. Both begin with
  position zero already accepted. Stop the other branch when it cannot extend the
  limiting prefix. Serve the minimum contiguous prefix, plus the regenerated
  rejection action when its position immediately follows that prefix; equal
  branch prefixes prefer the temporal replacement.
- Official output padding is not accepted execution. Only `accepted_exec_len`
  is served; the unused native-shaped tail never extends the real prefix.

Both verifiers are compulsory in this scope. A disabled reference uses native
inference, not a candidate eligible for an A³ result.

## Native CogACT adaptation

CogACT supplies sixteen 7D predictions and an epsilon-prediction DDIM head rather
than GR00T's flow head. The default maximum prefix is sixteen (the official example
uses five on its own checkpoint). H8 remains the benchmark's native comparator.
Weights, image processing, CFG pairing and action conversion remain native.

Known normalized actions are constrained before AND after every original DDIM
transition using `sqrt(alpha_bar)*fixed + sqrt(1-alpha_bar)*original_noise`.
At the clean endpoint they are fixed exactly. This maps the official noisy-path
conditioning to CogACT's trained noise coordinates; it is not a clean-action clamp
at noisy times, nor a claim of GR00T numerical identity. An empty mask reproduces
the native sampler including discarded RNG draws. Draft and verification share
the same DDIM budget; no separately quantized verifier is introduced. The default
retains all ten native steps. Smaller budgets select evenly spaced rounded indices
of the native grid while retaining its original first/last noise endpoints; four
steps use training indices 90, 60, 30, 0. This exposes the official step-budget knob
without changing the H8 native reference or composing a separate solver method.

Confidence and strict six-axis tolerances use native denormalized execution
commands, excluding gripper in comparisons, as the released code does. The actual
gripper is retained in sampled/replacement commands and converted to LIBERO once.
Default K=8, alpha=0.8, epsilon=1e-8 and axis thresholds follow the official GR00T
example. One global threshold scale and verification-block size are configurable.
They are not fitted from test trajectories or task-specific success outcomes.

Every candidate, conditional verification row, CPU transfer, selection, sampling
step and synchronization counts toward full-cost timing. No acceleration or
accuracy gain is presumed. The interface changes neither the native predicted
length nor the shared evaluation harness.

OFT's deterministic L1 head has neither independent diffusion candidates nor
conditional denoising and is not registered. The OpenPI code variant (whose
confidence ignores alpha), GR00T retraining, paper W4A4 verification, clustering
and strict paper-only acceptance are excluded. Do not mix these into this ID.

## Validation

CPU tests cover confidence ties, strict thresholds, confidence-order holes,
replacement priority, tentative masks, nonfinite inputs and native noisy-path
constraints. GPU tests require caller-provided model/trace resources and check
native/disabled parity, one shared VLM pass, active verification accounting,
reset determinism, real execution horizons and reversible detach. A separate
pinned-author comparison checks mask and decision parity without importing the
author model stack. Measurements are valid only after the common smoke, Dev-200
and four-prefix timing; the result ledger owns measured status.
