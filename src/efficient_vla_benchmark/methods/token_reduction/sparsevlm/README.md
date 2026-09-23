# SparseVLM

Paper-based implementation of the original [SparseVLM v1](https://arxiv.org/html/2410.04417v1),
equations 6-11. This is **not** SparseVLM+ or the released v1.5 fixed-budget variant.
The [author v1.5 reference](https://github.com/Gumpest/SparseVLMs/tree/734d9003879689713a132b83297dfad060a5c9af)
uses predefined budgets; substituting those would omit the original rank-adaptive claim.

Before the decoder, select text raters by mean visual-to-text softmax relevance
at or above the mean. At configured layers, inspect rater-to-visual attention,
compute its FP32 numerical rank and remove
`floor(rank_scale * (N_visual - rank))` low-score patches. Recycle the most
significant configured fraction of removed tokens, use kNN density-peak centers
and cosine assignment, then reconstruct each cluster by **sum** (Eq. 11).

Nonzero fractional recycle/cluster counts round down with a minimum of one.
Density uses log space to avoid exponential underflow; no global RNG is touched.
Ties follow native positional order. Every center owns itself to prevent empty
clusters for coincident features. Sum reconstruction is intentional, not a silent
switch to the averaging operation in some released implementations.

OFT raters exclude proprio/action placeholders; both cameras share the visual pool.
All nonvisual rows and original positions are protected. A cluster is anchored
at its latest member, preserving causal information order for the subsequent
CogACT port. No contiguous renumbering of action slots is allowed.

The pruning layers, rank scale and rank tolerance are explicitly configured
benchmark settings, not claimed author-tuned VLA settings. Default rank scale
0.25, recycling 0.3 and clustering 0.1 are fixed before LIBERO testing.
All rank, probing and clustering overhead is included in timing.
