# TEAM-VLA

Independent, training-free implementation of [TEAM-VLA](https://arxiv.org/abs/2512.09927v1), Sections IV-B/C. The [project repository](https://github.com/Jasper-aaa/TEAM-VLA) did not contain algorithm source at audit time. This is a paper-based port, **not verified reproduction of the authors' implementation or published scores**.

## Mechanism

Before layer 0, select each language token's most cosine-similar visual patch, expand its spatial neighborhood, and union with interval-random context samples. Process the two 16×16 camera grids independently. After zero-based layer 15, select the most text/action-relevant visual anchors and softly merge remaining visual features into those anchors. No extra vision pass, auxiliary model, temporal cache, or training is used.

## Explicit interpretation choices

- Text means OFT's prompt-language span, excluding BOS, proprioception, action placeholders and final stop. Middle-layer queries include that span and action placeholders; top-M ranks each visual patch by maximum query cosine similarity. The paper does not specify this top-M aggregation.
- Equations 3/4 use strict inequalities: neighborhoods exactly at the density threshold do not expand. Sparse expansion samples simultaneously from the original mask, at most one unrelated neighbor per sparse center. Context sampling chooses one token from each equal-width sequence interval and unions it with foreground; the final count is not a fixed retention ratio.
- Equations 5/6 transpose notation inconsistently. Follow the prose and declared `N_target × N_source` shape: each target distributes softmax mass over sources. Equation 9 is interpreted as division by `1 + received_soft_weight` (Equation 8), not the source-set cardinality. RMS normalization has no additional learned weights. Scoring and merging accumulate in FP32, returning features in model dtype.
- Keep each source's original patch/RoPE position and protect all nonvisual tokens and native OFT action readout. No position averaging or compression is applied.
- The uniform preset uses context ratio 0.25 and 80 merged tokens (paper's Goal/Long setting), not per-suite tuning. Kernel 3 and density threshold 1 follow Section V-D3; Section V-B3 instead says threshold 2. Layer 16 is interpreted as after zero-based layer 15.
- Random sampling has an instance-local generator, reset from episode seed plus config seed; it does not change global RNG state. Disabled or context=1/merge=512 bypasses the adapter entirely.

## Validation and use

Select `configs/experiments/openvla_oft/team_vla.yaml` through the shared CLI; supply resource paths on the command line. Tests cover spatial expansion, interval sampling, weighted aggregation, identity, real-model pruning/merging, reset and detach. All selection, feature mixing, bookkeeping and transfer overhead is included in policy timing. Four smoke episodes are connectivity checks, not formal success-rate evidence.
