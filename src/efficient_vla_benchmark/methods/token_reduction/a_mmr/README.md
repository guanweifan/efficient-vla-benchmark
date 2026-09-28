# A-MMR current-frame core

Independent implementation of sections III-A/B of
[History-Conditioned Spatio-Temporal Visual Token Pruning](https://arxiv.org/html/2603.06480v1).
The selectable ID is `a_mmr_current_frame`, not the complete navigation method.
The current-frame adaptive saliency/diversity selection is a central independently
defined contribution. Historical memory and query-guided history reweighting are
explicitly excluded: neither native policy consumes that navigation memory.

Following the methodology text, compute CLS-to-patch **cosine similarity**, not a
Q/K attention map. Normalize it with equation 1 (epsilon 1e-6). Seed with maximum
importance; subsequently choose the unselected patch maximizing importance times
(1 - maximum cosine similarity to selected patches), equation 2. Negative cosine
values remain negative. FP32 cosine computation and stable earliest-position ties
are explicit numerical conventions. Zero-range saliency gives stable ties, not a
new diversity-only fallback. No fixed semantic/diversity split or token merging
replaces the multiplicative selection rule.

Both native backends contain a genuine CLS-bearing DINO branch. Inspect its
second-to-last block, exactly where their unnormalized native patch features are
read. Row zero supplies CLS; all prefix/register rows are excluded from patch
selection. Do not fabricate a SigLIP CLS by averaging patches. Per camera, select
floor(256 times keep_ratio) patches, minimum one, using DINO features; retain the
same original patch memberships in the fused DINO/SigLIP projected sequence.
Using the native DINO branch is a disclosed dual-encoder transfer convention,
not a reproduction of the original navigation encoder or benchmark scores.

Drop selected-out rows immediately before LLM layer zero. Both vision encoders
and the dense pointwise projector still execute; no projector speedup is claimed.
Original masks/RoPE and nonvisual/action/cognition rows remain untouched. Native
heads, DDIM/CFG/RNG, observation preprocessing and H8 cadence are unchanged.
All feature copies, cosine matrices, greedy iterations and masks count in timing.
One snapshot per camera is cleared after selection/reset/close. Disabled or full
retention installs neither probe nor decoder adapter. No learned weights change.

Tests cover independent NumPy equations, negative similarities, ties, per-camera
budgets, protected rows, missing-prefix/feature errors and hook lifetime. Real-model
tests verify native probe/patch alignment, dense/no-op parity, active/reset/RNG and
detach on both separately installed backends. Common smoke, Dev-200 and compact
timing are required before reporting accuracy or speed.
