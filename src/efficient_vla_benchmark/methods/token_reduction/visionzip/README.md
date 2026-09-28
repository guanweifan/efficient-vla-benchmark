# VisionZip

Independent training-free implementation of [VisionZip](https://arxiv.org/html/2412.04467v2),
sections 2.3/A.2 and algorithms 2/3. The inspected Apache-2.0
[official revision](https://github.com/JIA-Lab-research/VisionZip/tree/8f86b55c6f000eb033e6912538af2dd7dcb30502)
is dated 2025-07-21. Its `visionzip/clip_encoder.py` and `utils.py` establish uniform
residual target spacing, mean-head key metrics and contextual aggregation. Reference
files are neither vendored nor imported; no new policy, encoder or training is used.

Both backends contain native SigLIP without CLS. Use the paper's explicit non-CLS
variant: softmax penultimate-block QK scores, averaged over heads and **received**
query attention. Select dominant patches, retaining original order. The remaining
patches also keep original order; choose contextual targets every floor(R/M) rows,
then assign non-target residuals by cosine of mean-head native keys. Default
dominant fraction 54/64 reproduces the paper's 108/20 and 54/10 budget splits.
Retain floor(patches times keep_ratio), minimum two; both contributions remain active.

Follow the **released aggregation**: contextual feature = target feature + mean of
assigned non-target features. This is not a pooled average including the target.
Empty groups retain their target. The paper's abbreviated average-merge pseudocode
does not spell out this weighting; the implementation/metadata disclose the code
convention explicitly. Stable ties and epsilon 1e-12 zero-norm handling are stated
numerical conventions. Score and aggregation arithmetic is FP32 with autocast
disabled locally, not a change to native model precision.

Use the same SigLIP-derived grouping on the complete fused DINO/SigLIP features,
independently for each camera. Merge **before** the nonlinear native projector;
only compressed rows pass through that original pointwise MLP. Scatter projected
outputs into a shape-only native assembly buffer and remove its unused rows before
LLM block zero. Do not merge projected features or independently mismatched encoder
groups. All grouping, projection, scatter and selection cost belongs in timing.

Each merged group occupies its latest original member's position; sort the resulting
dominant/contextual union by those anchors. This native-position transfer preserves
causal information order and RoPE/masks, unlike the reference's CLIP compact sequence.
No CLS row is invented or added to these native patch-only model inputs. Native
vision encoders, preprocessing, action/cognition readout, DDIM/CFG/RNG and H8 cadence
remain unchanged. Report this common-checkpoint transfer, not original VLM scores.

Disabled/full-retention modes install no hooks. Per-query evidence and projected
row mappings clear on reset/error/detach; instance-only wrappers restore the native
projector. Tests cover independent grouping/aggregation, latest-member anchors,
nonlinear projection order, protected rows, autocast and real-model dense/reset/RNG
parity. Common smoke, Dev-200 and paired replay timing are separate required evidence.
