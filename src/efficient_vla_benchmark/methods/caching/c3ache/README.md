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

Use `c3ache_conditioned_residual.yaml` for the selected report configuration:
six live blocks, two cached steps, refresh interval two. Cross-chunk reuse remains
active. High residual cosine similarity alone does not bound output-head error;
this native conditioning boundary requires closed-loop validation. See
[results](../../../../../docs/benchmark_results.md) for selection and measured outcomes.
