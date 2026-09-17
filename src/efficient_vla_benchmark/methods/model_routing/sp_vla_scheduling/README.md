# SP-VLA scheduling core

Category: model_routing (survey §5.3, cross-reference §5.5).
Inference-only on the unmodified joint-suite OpenVLA-OFT checkpoint.

Scope: independently port the paper's model scheduling contribution. Token pruning
is excluded, not composed with another method. This is not full SP-VLA reproduction.

- [Paper](https://arxiv.org/abs/2506.12723), version 3.
- [Official source](https://github.com/ChildTang/SP-VLA/tree/61872a4d6d8e9c18f08b6474ceca2ff2fe08e942), MIT.
  Reference functions: `fit_next_action`, `save_actions`, `predict_action` in
  `prismatic/extern/hf/modeling_prismatic.py`. This is a small independent NumPy
  implementation, not an executable dependency on the author's entire runtime.

At an empty action queue, inspect the recent executed Cartesian commands. If their
vertical motion is sufficiently small and at least four of the last six commands
came from the model, fit a ridge trend and emit one replacement command. Fit XYZ;
hold rotation/gripper and bound the extrapolated change by recent deltas, exactly
as the pinned code does (the paper's prose also mentions rotation fitting).

OFT adaptation: keep its native eight-action query/executed chunk. Test the gate only
when that chunk has finished; a fitted action lasts one step, after which the gate
is evaluated again. Never pad a fitted action into eight repeats or force the
baseline to query every step. Only executed native actions count as model history.

The common preset uses the author's spatial gate (ratio 0.4, Z 0.3, one gate step)
for all four suites. It is an explicit untuned cross-suite preset, not the paper's
suite-specific settings. All thresholds/window/fit parameters live in YAML.

Use `configs/experiments/openvla_oft/sp_vla_scheduling.yaml` through the same evaluate
and benchmark commands as baseline. Set `enabled: false` for a native control no-op.
Requires control replay; query/prepared-only timing cannot execute this mechanism.
Report actual query count, replacement count and total policy time/per-control-step
latency against native horizon 8. Fixed-trace gains say nothing about success loss;
planar extrapolation may fail near contact or changes of direction.

The NumPy method is independent of OFT. Future bindings must verify action units,
XYZ ordering and history semantics, then declare the relevant action capability.
