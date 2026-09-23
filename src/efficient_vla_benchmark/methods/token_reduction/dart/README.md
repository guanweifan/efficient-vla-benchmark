# DART

[Duplication-Aware Reduction of Tokens](https://arxiv.org/abs/2502.11494), not the
similarly named speculative-decoding method. Reviewed author revision
[`0c888673198b0f2784053a5686f95a519ceef209`](https://github.com/ZichenWen1/DART/tree/0c888673198b0f2784053a5686f95a519ceef209).
Independent implementation; the reference source is Apache-2.0.

Before decoder layer `pruning_layer`, use the preceding layer's post-RoPE keys
to choose visual and language pivots by L1 norm. Apply the native final RMSNorm
to that layer's hidden output for cosine comparisons. For each pivot, retain
the least-similar remaining visual tokens, removing them from the candidate set.
Preserve visual pivots, remove language pivots only from the visual selection
set (all original language tokens remain in the sequence), and sort by position.

The released budget includes extra visual pivots: default OFT 512 patches yield
260 retained rows, not 256. Pivot processing uses ascending native positions
rather than implementation-dependent Python set iteration; ties are positional.
Scores use FP32. OFT language pivots exclude proprioception and action placeholders.
Its two views share a global visual budget. Native masks, RoPE and action slots
are protected; this is a controlled VLM-to-VLA transfer, not author VLA-code parity.
No attention probabilities, training or calibration are required.

## CogACT binding

CogACT probes the preceding native attention module's post-RoPE keys without
replacing SDPA. Language pivots exclude the final cognition EOS. The same shared
rule retains 132/256 visual rows by default; cognition and diffusion RNG remain
native. Device-side pivot gathering avoids implicit scalar host synchronization.
