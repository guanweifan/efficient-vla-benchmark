# ST-Merge: spatial VLA path

Paper-based port of [ST-Merge](https://arxiv.org/html/2606.29350v1), section III-E's
single-frame multi-view VLA deployment, not its video-Qwen path. The linked author
repository returned 404 during integration. No claim of verified source parity.

Merge inside shallow vision blocks, using Gaussian spatial locality, key dot-product
similarity, multi-queue matching and size-weighted averaging. As in the paper's
SigLIP VLA path, no video history or vision RoPE correction is introduced.

Explicit reconstruction conventions: row-major interleaved queues; each anchor
proposes one candidate per other queue; duplicate source proposals retain the
highest score, ties are stable; top-scored unique sources merge into surviving
anchors. Hence collisions can retain MORE than the nominal token budget. No forced
extra merges from a different algorithm. We do not claim O(N) complexity for the
pairwise matching implementation or borrow the paper's high-resolution speedup.

The matching score follows Eq. 2 literally, without additional key L2 normalization.
Earlier runs used cosine similarity; those results remain historical evidence and
do not describe this corrected scoring path. The multi-queue collision policy and
surviving-anchor convention remain unverified: Sec. III-A describes one-to-many
matching, whereas Eq. 4 describes source-to-target fusion. Do not present this
anchor-preserving OFT port as an exact reconstruction of all author choices.

OFT adaptation: run SigLIP first using the unchanged native timm forward, derive
one per-view mapping and apply the SAME mapping to DINOv2 at the selected block.
Preserve DINO prefix/register tokens. Restore sparse outputs into original patch
slots for the native projector and remove vacant rows before LLM layer 0. Thus
vision blocks and the LLM see fewer tokens, but the projector still processes the
full grid; this bridge overhead is included in measured latency. Surviving anchor
positions preserve native LLM RoPE and action-head slicing. Both cameras remain
separate; neither receives the other's tokens.

Default layer 1, four queues, sigma 2 patches and target 128 tokens per view are
untuned OFT port parameters, not verified author settings. Disabled/full-retention
mode installs no hooks. Reset and close clear instance-local grouping/probes.

Use `configs/experiments/openvla_oft/st_merge.yaml`. Report actual per-layer vision
and language token counts, complete overhead, failures and original input 224
resolution. Do not compare against the paper's trained high-resolution baseline.
