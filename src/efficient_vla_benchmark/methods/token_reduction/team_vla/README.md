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

The 2026-09-21 re-audit inspected the author repository at
[`a7d6e373`](https://github.com/Jasper-aaa/TEAM-VLA/tree/a7d6e373a1e2a8d6c37d12a26d44c232a73fda73):
README and illustration only, no algorithm code. Independent neighborhood-union,
RNG-isolation and received-mass tests supplement the scalar Eq. 5-9 reference.
No evidence justified replacing the existing softmax/aggregation interpretation
or tuning the preset to the four smoke tasks. Poor smoke outcomes therefore remain
observations of this disclosed port, not proof that the published method fails.

## CogACT fidelity boundary

The 2026-09-22 recheck finds the same source-only project status and defers the
complete CogACT port. Early foreground expansion/context sampling are portable,
and implementing visual feature merging is ordinary engineering. The unresolved
part is Section IV-C's joint text/action-token guidance in the shared LLM space:
native CogACT has no LLM action slots; its separate DiT actions use another
representation and receive only one compressed condition. Treating cognition as
the missing action-token set or projecting DiT features back would introduce an
unverified guidance rule. We do not label the early stage alone as full TEAM-VLA.
This is a fidelity deferral, not proof that a new CogACT variant cannot be designed.

## Bounded development diagnostics

The Dev-200 diagnostics isolate early sampling (`team_vla_context50`: 0.25 to
0.5 context ratio) from middle-layer compression (`team_vla_merge160`: 80 to
160 anchors). All expansion, scoring and weighted-merge equations stay fixed.
These are parameter comparisons, not alternate interpretations chosen by success.

The fixed Dev-200 comparison scored 97/200 by default, 82/200 with context50,
and 167/200 with merge160, against native 198/200. Increasing middle-layer
anchors helped substantially; increasing early context alone did not. Do not
combine the two variants without a separately scheduled experiment. The remaining
accuracy gap and paper interpretation uncertainties remain explicit.
## Final-readiness boundary (2026-09-25)

The OFT port remains pending core clarification. Eq. 5–9's soft-weight direction
and cardinality-versus-received-mass normalization affect the merged feature,
not just an incidental hyperparameter. Increasing the anchor count cannot settle
that interpretation. Retain the port diagnostics without certifying a faithful
negative result or scheduling it as a confirmed primary method.

The final source recheck still finds only README/assets at the pinned author
revision. The missing joint text/action top-M aggregation is also a core issue,
independent of the merge-direction/normalization ambiguity. No success-guided
alternative interpretation is substituted. CogACT's absent LLM action-token set
does not justify silently replacing that guidance with cognition-token scores.
