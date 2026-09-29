# ETA-VLA / ILSA core

Independent implementation of [ETA-VLA section III-C](https://arxiv.org/html/2603.25766v1#S3.SS3),
equations 3--8 and Algorithm 1. Only the Intra-LLM Sparse Aggregator is supported;
the trained temporal fusion module and driving trajectory optimizer are excluded.
This is a manipulation transfer of an explicitly scoped driving-VLA contribution,
not a complete ETA-VLA reproduction or a published NAVSIM-score claim.

At the selected layer, compute visual-to-language softmax similarity and retain
language anchors with strictly above-mean alignment. Native unrotated Q/K give
bidirectional, RoPE-free semantic attention; normalize over ALL current keys and
average anchor rows and heads. Global top-K semantic selection is supplemented
by the most diverse remaining tokens, using Eq. 7's average cosine to every other
current visual token in the same camera view. Recycling is global across rejected
tokens, not a forced equal camera quota. Neither component is disabled for speed.

## Native transfer and explicit conventions

H is the native normalized pre-attention input, where native Q/K are projected.
Semantic probes are extra computation: native causal/bidirectional attention and
RoPE remain unchanged. OFT excludes proprioception/action placeholders from text
anchors; CogACT excludes its final cognition EOS. All nonvisual output rows and
original token positions are protected. Select the current block's output after
its MLP; only subsequent blocks become shorter. No current-block MLP savings or
linear-time full layer are claimed. Q/K projection, scoring and selection costs
are included in replay. CogACT's single view remains valid for within-view diversity.

Table III motivates layer 4 (zero-based 3), 65% retention and a 90/10 semantic/
recycle split. These are one common engineering preset, not fitted per-suite.
The retained count is floor(current visual count times keep_ratio), at least two;
round the recycle count down but retain at least one token from each component.
Scores are FP32 with local autocast disabled; stable original-position ties and
cosine epsilon 1e-12 are declared numerical conventions. Singleton-view diversity
is zero. Eq. 7 uses all view members, while Algorithm 1 abbreviates the eligible
recycling input; candidates are only semantic rejects. No learned scorer is added.

If strict above-mean selection yields no text anchors, Eq. 6 is undefined. The
explicit safety boundary is dense native execution, with a diagnostic flag; no
uniform or visual-only substitute is passed off as ILSA. Disabled/full-retention
modes install no hooks. Ephemeral selections clear on reset/detach; weights,
preprocessing, native action head, DDIM/CFG, RNG and H8 cadence are unchanged.

Registration is not certification. Independent equation tests, native boundary
tests, four-suite smoke, Dev-200 and paired replay are separate acceptance gates.

Initial validation passed 18 independent equation/config/lifecycle tests and one
real-model parity/active/reset/detach test on each backend. Four-suite smoke
completed 4/4 successes with zero errors on each backend; this is a connectivity
check, not a formal accuracy or speed claim. Dev selection remains pending.
