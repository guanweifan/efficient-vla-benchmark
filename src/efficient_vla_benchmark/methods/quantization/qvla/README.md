# QVLA runtime feasibility components

Status: **not registered, not a completed reproduction, not a formal result row**.
These components establish mixed-channel packed execution before calibration and
backend admission. No experiment YAML or catalog metadata is intentionally exposed.

Reference: [official allocation source](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/qvla/assign_gates_from_sensitivity.py).
`allocation.py` implements the absolute lower-bit sensitivity cost per saved bit,
with stable candidate insertion order. It is not marginal-error demotion. The
budget is channel-average bit width, not parameter-weighted model bits/weight.

`packed.py` groups output rows by 0/2/4/8/16-bit assignment and restores their original
order. Zero-bit rows retain bias, 16-bit rows retain native floating weights, and
2/4/8-bit rows use shared packed kernels. Bias is included before output rounding.
Non-grouped zero-padded Conv2d uses unfold plus the same row execution; unfold,
scattering and all precision conversion must be included in future timing.

The initial OFT diagnostic uses a checkpoint-matched handover allocation across
430 vision/language modules and verifies packed outputs against an independently
constructed dense quantization oracle. That allocation used a different greedy
cost; it is retained **only as a runtime layout fixture**. Scores were not saved,
so the official allocator needs fresh training-only calibration before admission.
Do not relabel these gates as official-code allocation or report this smoke as
formal QVLA accuracy. The oracle is not a speed competitor or production fallback.

Tests cover absolute-versus-marginal allocation, mixed and zero-only precision,
all-native exact identity, bias, output order, convolution and same-topology reload.
The GPU model probe/episodes remain ignored diagnostic evidence. CogACT artifact
availability does not establish tested CogACT runtime support.

Remaining gates: calibration/provenance, supported target validation, irreversible
method lifecycle, backend bindings/configuration, fresh reload, compact replay,
then the owner's Dev/final schedule. Do not infer paper-wide support from primitives.
