# SparseVLM+ (V2.0)

This port follows the author's [V2.0 branch](https://github.com/Gumpest/SparseVLMs/tree/9e1f649ef6a78878e2b10d1d6f49ecd7f4d84094)
and its `USE_VERSION=2_0` execution path. Both native backends use the same
selection equations; this replaces the former paper-v1 rank/recycling port.

Before the decoder, select prompt raters with strict above-mean visual-to-text
softmax relevance. At layers 2/6/15, divide each head's rater-to-visual
AFTER-softmax probabilities by the corresponding attention between all-ones
queries/keys with native RoPE. Select the 14 heads with greatest corrected
text-to-visual mass, average their scores and retain the highest-ranked patches.
The attention denominator includes all native keys; the correction affects
selection only. Native SDPA output is unchanged before pruning.

Official V2.0 budgets for a 576-patch input are:
192: [300, 200, 118]; 128: [238, 108, 60];
96: [246, 54, 28]; 64: [66, 34, 20].
Profile names are effective budgets, not final counts. Multiply each stage budget
by original native visual count / 576 and the explicit `budget_scale`, then floor
with minimum one. OFT's two views form one 512-token pool; CogACT uses 256.
Cap each stage at current count minus one, matching the reference top-k rule.

V2.0 disables recycling and preserves original positions. There is no matrix-rank
calculation, cluster reconstruction or v1.5 position renumbering in this port.
Native text/proprio/action or cognition rows remain protected. Raters exclude
non-language placeholders and CogACT cognition EOS. Entirely tied rater relevance
is rejected explicitly rather than inventing another scoring rule.

The native causal/padding or OFT bidirectional mask remains authoritative.
Selected per-head rectangles are algebraically the same as the author's full
correction matrix; their extraction avoids allocating unused full-query scores.
Finite scores are required; no undocumented epsilon/clipping changes correction.
The 14-head setting, profile and native-count mapping are disclosed migration
parameters, not author-tuned LIBERO results. Every probe, selection and correction
cost is included in full-cost timing. Reset/close release all transient probes.

## Selected Dev configurations

Use `configs/experiments/openvla_oft/sparsevlm_v2_scale0875.yaml` (profile 192,
scale 0.875) or `configs/experiments/cogact/sparsevlm_v2_128.yaml` (profile 128).
These give 194/200 and 181/200 on frozen Dev-200, respectively. More aggressive
tested settings cross the native-relative 3-percentage-point loss boundary.
They are the fastest eligible measured candidates, not a global optimum.
See `docs/benchmark_results.md` for paired full-cost timing and evidence identities.
Old paper-v1 Final scores do not apply to this V2.0 implementation.
Both selected configurations now have one frozen Final-2000 run. CogACT's
full-set loss exceeds the Dev selection target; keep the measured outcome and
the original Dev evidence rather than retuning from Final results.
