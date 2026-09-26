# Reducing Temporal Redundancy: visual core

Independent [section III-B](https://arxiv.org/html/2607.12287v1#S3.SS2) implementation.
The ID temporal_redundancy_visual implements a principal visual-reuse contribution,
not the paper's full system or trained two-step flow generator (section III-C).

The first and alternating policy queries compute native full visual attention.
Intervening queries rank first-layer patch keys by change from that dense reference,
update the least-similar fraction, and reuse the remaining attention outputs/KV.
Every later block uses the same per-view/encoder selection. All residual and MLP
rows still compute; the visual grid, positional embeddings, native penultimate
readout and projector stay intact. This is attention-row reuse, not token deletion.

Both fused vision branches preserve their own feature/key spaces. Each camera and
encoder has a separate cache; prefix/register tokens always update. Flattened
multihead normalized keys define cosine; ceil determines retained patch count;
stable row-major ties are explicit adaptation conventions. Full refresh calls
execute original attention and capture QKV. Reuse uses native fused/manual attention
for the selected rows against the complete indexed key set. All overhead is timed.

OFT binds two cameras; CogACT binds one. The shared method never imports either
backend. Disabled/update_ratio=1 leaves all native forwards untouched. Reset clears
all episode/instruction history; close restores instance forwards. The checkpoint,
LLM attention/masks and action sampler are not modified. No training is required
for this visual-only core; no claim is made for the omitted trained action branch.

The report selects update_ratio=0.50 on OFT and the explicit
`temporal_redundancy_visual_update75.yaml` preset on CogACT. CogACT's bounded
retention adjustment improved Dev-200 from 168 to 180 successes; OFT retained
192 successes. Both measured ports include all cache/indexing overhead and
were slower than their paired native references in the compact workload.
These are scoped implementation/backbone results, not a rejection of the
paper's full system or a claim about its excluded trained action generator.
