# EVO: Evolving Cache Schedules

[Paper](https://arxiv.org/html/2607.20293v1),
[pinned reference](https://github.com/pillom/EVO/tree/09e0fcbdba14ebf7d8180f434dd5a015aeb0a71e).
This independent implementation retains global block–timestep budget allocation,
rollout-only evolutionary fitness, redundancy-guided plus random initialization,
tournament/crossover/mutation/repair/elitism, canonical deduplication, and a
baseline-relative quick trigger followed by independent confirmation.

CogACT supplies 24 attention/MLP residual branches, not DP-T's extra cross-attention
branches. All residual inputs, norms, CFG rows, native DDIM10/RNG and H8 remain
native. Caches reset at each model query. First and final steps refresh every
branch, matching the reference runtime safeguard with its hard-coded 99 generalized
to T-1. Those endpoints are INCLUDED in the actual global budget, not unreported
extra work. Interior allocation remains global and unrestricted between branches.

Offline search uses three disjoint frozen placement groups: native reference plus
activation-prior capture, quick candidate evaluation, and independent confirmation.
Every placement excludes all fifty standard states per task and other groups;
neither Dev-200 nor final outcomes enter the search. The prior averages all-pairs
cosine dissimilarity (Eq. 6), then uses the reference min-max normalization and
soft sampling floor (Eq. 7). Two native queries per reference episode contribute
features; their collection changes neither predictions nor RNG. The final choice
depends on rollout fitness, not this feature prior.

The explicitly authorized campaign is bounded by 640 extra episodes, INCLUDING
the 40-episode native reference, and at most 24 candidates. Each candidate quick
test has 20 episodes; independent confirmation has 40. A compact population of
four with two elites explores global budgets 168/120/72 out of 240 branch calls;
these are candidates, not selected report settings. Reserve each job before launch
and never refund a failed allocation or silently retry it. Search/source identities
and every success/failure remain in the artifact audit. Budget exhaustion is NOT
claimed as convergence or target attainment. One layout is shared by all suites;
the paper's per-task schedules and original DP-T scores are not imputed.

The deployment binding requires a checksummed frozen artifact produced after
rechecking all population, source, layout, outcome and budget identities.
Confirmation failures remain explicit. No optimizer or feedback runs at deployment:

```bash
evb evaluate --config configs/experiments/cogact/evo.yaml \
  --model-path "$MODEL_DIR" --artifact-path "$EVO_ARTIFACT" \
  --libero-root "$LIBERO_ROOT" --output-dir "$RUN_DIR"
```

Only after all candidate layouts are frozen may Dev-200 and the unchanged compact
replay protocol select/report a configuration. Independent search cost is separate
from steady-state latency; deployment overhead is inside the common timer.
OFT L1 is unsupported because it has no iterative denoising lattice.
