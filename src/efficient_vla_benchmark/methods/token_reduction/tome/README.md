# ToMe

Independent equation implementation of [Token Merging](https://arxiv.org/abs/2210.09461),
a general vision baseline transferred to the two native VLA vision encoders.
The [released reference](https://github.com/facebookresearch/ToMe/tree/af95e4b1befa172dadccd8c81e223b10090f9579)
is CC BY-NC 4.0; it is inspected for mechanism comparison, not vendored or imported.

The core remains intact: alternate-index bipartite matching of cosine-normalized
mean-head keys, select the r strongest source-to-destination edges, size-weighted
merging BETWEEN attention and MLP, and log(token mass) proportional attention in
subsequent blocks. r is a count per block, capped at half the non-prefix tokens.
CLS and all DINO registers remain protected. For zero/one prefix the released
unmerged-A/B order is preserved because it affects the next block's partition.
For multiple prefixes, keep all prefix rows first in native order and exclude
them from both sides of matching; patch parity still uses full-sequence indices.
Exact score ties use stable index order. Key matching uses FP32 cosine scores.
Native LayerScale, pre-normalization and residual paths remain present. Fused SDPA
uses an additive log-mass bias instead of materializing a full attention matrix.

DINO and SigLIP independently choose groups for each native camera. Their sparse
penultimate outputs are unmerged using ORIGINAL patch membership before native
feature concatenation. This preserves spatial correspondence even when the two
encoders choose different groups/counts. Do not concatenate unrelated sparse rows
or claim the unused members have their original unmerged feature values.

This bridge accelerates vision blocks only: the projector, full visual LLM context,
native action readout, preprocessing, sampler and H8 cadence stay unchanged. It
does not compose ToMe with an extra language-token pruning algorithm. Native timm
still executes its final unused block; merging ends at the served penultimate
readout so source membership cannot drift after that readout is saved. All merge,
proportional-attention and unmerge costs belong in common full-cost timing.

Disabled/r=0 modes install no wrappers. Native prefix tokens are never merged
with image patches or one another. State is per
encoder call and clears on errors, reset and idempotent detach. Tests compare
matching/group mass and proportional attention to independent references, verify
the native penultimate unmerge boundary and check real-model reset/no-op/detach.
GPU tests take caller-supplied resources. Shared smoke/Dev/replay protocols apply;
less vision computation is not a promise of end-to-end acceleration.
