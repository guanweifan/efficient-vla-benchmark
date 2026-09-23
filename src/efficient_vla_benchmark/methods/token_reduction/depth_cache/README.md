# DepthCache primary-view core

Implements the principal contribution in [DepthCache](https://arxiv.org/html/2603.10469v1),
Sections III-B1/B2/B3, under the distinct ID `depth_cache_primary`.
No public author implementation was found during this audit.

Included: warmup semantic protection plus depth-edge protection; K-means depth
regions and depth-proportional budgets; cross-query gradual bipartite merging;
persistent group membership; current-frame size-weighted reconstruction; and
depth-change-triggered region restoration/re-convergence.

**Excluded by explicit scope:** Section III-B4's object-carrying-dependent scene
reinitialization and Section III-C's anticipatory wrist-view controller. The paper
does not specify a verifiable carrying detector or how an as-yet-uncomputed chunk
controls compression of its own input. Wrist tokens remain full-resolution on OFT.
This must never be reported as full DepthCache or credited with its dual-view results.

Use measured metric depth, aligned with the primary RGB stream. LIBERO supplies
depth only with the explicit `evaluation.depth: true` setting. RGB-only historical
traces are rejected, never augmented with synthetic depth. Compare timing against
a baseline replaying the **same** depth-bearing traces; depth conversion, transfer
and all selection work remain visible. Simulator rendering itself remains outside
the existing policy-timing boundary, as it does for RGB.

Five warmup query frames, three regions, five-step progressive window and maximum
merge ratio 0.7 follow the publication. Unspecified choices are explicit engineering
conventions: final-layer language saliency, population standard deviation, nearest
metric-depth resampling and patch means/max edges, deterministic quantile-seeded
10-step K-means, region-mean depth normalization, and exposed metric thresholds.
They are fixed before LIBERO smoke, not tuned against success.

Merging progressively extends existing groups using fresh current-frame features;
it never averages stale features across time. Bipartite rounds handle budgets
above 50%. A merged row uses its latest original member as the causal-safe anchor.
A dynamic region immediately restores all its original rows and restarts its
progressive clock when the depth window becomes static. Reset clears all history.
