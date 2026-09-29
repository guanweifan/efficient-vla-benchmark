# VLA-IAP

Independent, training-free implementation of [VLA-IAP v1](https://arxiv.org/abs/2603.22991v1), Equations 1–14 and Appendix Algorithms 1/2. The [project repository](https://github.com/Chengjt1999/VLA-IAP) did not expose algorithm source at audit time. This is a **paper-based implementation with explicit unverified choices**, not a reproduction of the authors' checkpoint scores or 30/50/70% presets.

## Mechanism

Compute semantic similarity, second-order visual feature differences with temporal smoothing, and Sobel edge priors. Semantic/motion mask IoU selects conservative double-weak background removal or aggressive semantic-core/motion union. Finally union the selected set with high combined semantic/motion/edge priority. Prune before layer 0, preserving original positions and all nonvisual tokens. Each camera retains its own 16×16 grid. There is no learned scorer, additional encoder, attention probe, action-head change or method composition.

## Explicit interpretation choices

- The paper specifies temperature 0.01; the default uses its ablation values gamma=0.7, k=0.5, IoU threshold=0.05 and edge weight=1. These do not define the entire implementation.
- Unspecified values are exposed, untuned engineering choices: background k=-0.5, semantic-core radius=3 patches, combined-score threshold=1, pooling/closing/Gaussian kernels=3, Gaussian sigma=1 and epsilon=1e-6. No claim that these are the authors' values.
- The paper does not identify the OFT text encoder, pooling rule or exact centering axes. Use existing projected visual features and mean OFT prompt-language embeddings in the shared LLM space; subtract the same per-view visual spatial mean from both modalities, then L2-normalize. No external CLIP or additional weights are introduced. This alignment assumption requires future author-code verification.
- Thresholding, min–max normalization, morphology, semantic-center distance and IoU operate per camera, using population standard deviation. Semantic pooling excludes padded cells; Sobel reflects image boundaries; Gaussian smoothing replicates grid boundaries. Constant maps and empty unions are protected numerically. A fully empty view keeps its best combined-priority patch.
- Use the same cropped RGB images as the native vision encoder, average Sobel magnitude per 14×14 patch, and run all score arithmetic in FP32. The first two policy queries retain all tokens to initialize history. History holds two projected-feature arrays and one motion map; reset on episode/layout change.
- Temporal t denotes consecutive policy queries under native H=8 control, not every environment frame. There are no extra vision passes. All RGB conversion, Sobel filtering, history handling and selection costs are timed.
- Section 3 explicitly replaces fixed-ratio truncation with threshold/union selection, while its tables report fixed retention percentages without specifying their calibration. Follow the actual equations and report **measured retention**, not a fabricated fixed-ratio equivalent. More conservative or even slower outcomes remain valid benchmark observations.

## Use and limitations

Select `configs/experiments/openvla_oft/vla_iap.yaml` through the shared CLI, supplying model and environment paths externally. Disabled mode leaves native prediction unchanged. Tests cover equations, mask switching, geometric rescue, numerical cases, real-model warmup/reset/detach, and bounded history. Four smoke episodes do not establish formal success rate. Validate the missing conventions against future author code before citing this run as faithful numerical reproduction.

The 2026-09-21 author-source recheck at
[`3131851f`](https://github.com/Chengjt1999/VLA-IAP/tree/3131851f1b2c5cbd3e7a861a52062d717f4e0637)
still contains no algorithm implementation. Independent NumPy semantic-score
checks validate the disclosed centering/pooling convention, not its authorship.
Additional tests cover linear-motion cancellation and the conservative branch at
exact IoU-threshold equality. No new encoder or fixed-ratio top-k override is
invented to match the paper's performance table. Feature alignment and the missing
threshold calibration remain the main fidelity limits, despite executable equations.

## CogACT fidelity boundary

The 2026-09-22 recheck at the same author revision still provides no semantic
encoder/pooling implementation. Motion history, mask logic, geometry and Sobel
operations are portable. The blocker is narrower: Eq. 4's aligned visual/text
features define the semantic mask and therefore BOTH adaptive modes. Merely
matching embedding widths or copying OFT's unverified mean-embedding assumption
does not establish that construction for CogACT. The existing OFT implementation
remains explicitly provisional; it is not evidence that a faithful CogACT binding
has been verified. Keep CogACT unregistered until this core rule is grounded.
## Final-readiness boundary (2026-09-25)

The OFT port remains pending core clarification: the unidentified semantic
encoder/alignment/centering construction controls both adaptive selection modes.
Matching feature widths and passing equation tests do not establish semantic
alignment. Retain the disclosed port diagnostics, but do not use its success
loss as evidence that the published core fails or enter it as a confirmed primary.

The final recheck still finds no encoder/pooling implementation at the pinned
author revision. The renamed IAprune v2, dated 2026-08-26, is outside this review's
2026-08-17 cutoff and changes semantic centering/temperature, motion normalization,
mask budgeting and rank fusion. It is not a clarification that can silently
repair v1. Keep the in-scope v1 provisional rather than mixing both algorithms.
