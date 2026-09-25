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

## CogACT binding

`configs/experiments/cogact/st_merge.yaml` shares the same matching and fusion
implementation on one native 224px DINO/SigLIP grid. No action-head-dependent
saliency is introduced. The paired backend adapter accepts CogACT's native image
dictionary and preserves DINO's prefix/register tokens and penultimate readout.
The native dense projector bridge is retained; vacant visual slots are removed
before causal LLM layer zero, preserving original anchor positions and cognition
EOS. DDIM and global RNG are unchanged. Nominal 128 patches is a target, not a
guarantee when matching collisions leave additional anchors. The reconstruction
conventions above remain disclosed; this port does not resolve missing author code.

The fixed CogACT preset completed all four development episodes with zero runtime
errors but **0/4 successes**, versus native 3/4 in the same-source cohort. Do not
describe this configuration as accuracy-validated. Native no-op/reset/detach,
independent matching/mass checks and shared DINO/SigLIP membership checks passed;
the observed regression is retained, not tuned away. These checks do not prove
that every author convention is recovered or that the published method fails.
See the central ledger for its paired speed/memory measurements and run identity.

## Bounded development diagnostics

The bounded Dev-200 diagnostic presets `st_merge_keep192` and `st_merge_layer6`
change only the target budget (128 to 192 per view) or merge layer (1 to 6),
respectively, on both backends. Matching, mass weighting and paired DINO/SigLIP
membership are unchanged. These test compression strength and early-feature
sensitivity; neither resolves the paper's unverified grouping conventions.

On the fixed Dev-200 subset, the current-round OFT default, keep192 and layer6
presets scored 124, 197 and 174 successes; CogACT scored 34, 162 and 56 out of
200. Native references scored 198 and 183. Budget relief is the stronger tested
intervention, but CogACT still loses 10.5 percentage points at keep192. One
CogACT layer6 shard logged a simulator instability warning; its outcomes remain
included without assigning the buffered warning to an unverified episode.
These results diagnose this disclosed port, not all author conventions.

Budget relief also reduces acceleration: paired keep192 replay measured 1.023x
on OFT and 1.180x on CogACT, versus 1.151x and 1.425x for their default presets.
Near-native OFT success therefore does not imply a substantial speed benefit.
## Final-readiness boundary (2026-09-25)

Both bindings remain pending core clarification: Eq. 4's fusion direction and
the text's one-to-many matching do not uniquely determine the current surviving-
anchor/collision construction. Improved Dev success does not resolve that issue.
Keep the disclosed port and useful diagnostics, but do not schedule a final
benchmark row claiming a verified core or attribute its losses to the paper.
