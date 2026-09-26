# C3ache conditioned-residual core

[Paper](https://arxiv.org/html/2606.08962v1#S3): cache a separate DiT residual
for each sampling step across action chunks, refresh periodically, and fully
compute tail steps. This CogACT binding is an explicit boundary adaptation,
not the complete Fast-WAM implementation or its published scores.

CogACT concatenates cognition as a separate token. Its pre-block ACTION rows do
not contain the observation, and the final projection discards the cognition row.
Blindly skipping all blocks would prevent new observations affecting the result.
We therefore evaluate a native conditioning prefix (one block by default) on every
step and cache the aggregate remaining residual R = h_L - h_prefix.
On reuse, h_L = current h_prefix + cached R. This preserves cross-chunk residual
reuse and fresh-observation dependence without adding weights or another method.
The selectable ID is c3ache_conditioned_residual to keep the adaptation visible.

Default cache steps 0–6, refresh interval 4 chunks and three full tail steps on
the native ten-call DDIM schedule. The prefix length is exposed, but zero is
rejected. Sampling, CFG, RNG, normalization and action cadence remain native.
Cache entries retain both CFG rows; reset at episode/instruction changes and
errors. Disabled/interval-one modes attach no wrappers. No within-query
EfficientVLA cache is combined. Include live-prefix and bookkeeping overhead.

## Selected CogACT boundary

Use `c3ache_conditioned_residual_prefix6.yaml` for the selected report configuration:
six live blocks, two cached steps, refresh interval two. The initial one-block /
seven-step preset scored 1/200; reducing to two cached steps alone gave 79/200;
six live blocks recovered 183/200, equal to the matched native Dev count.
All candidates retain active cross-chunk reuse; none is a baseline no-op.
The initial and conservative presets are diagnostic configurations, not defaults
for the report. Their evidence remains separate from the one primary selection.

A two-trace native diagnostic found that high residual cosine similarity can
still yield large errors after the output head. This motivates reducing the
cached suffix, not changing DDIM or claiming the original Fast-WAM result.
The selected port's roughly 1% measured speed difference is not robust evidence
of a useful end-to-end gain under a single compact timing pass.
