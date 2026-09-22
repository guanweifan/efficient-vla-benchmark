# SP-VLA: scheduling and visual token pruning

The canonical method ID/directory is now `sp_vla`. The existing experiment file
`sp_vla_scheduling.yaml` remains a scheduling-only preset using this same method;
`sp_vla.yaml` enables both contributions. Old recorded run IDs are unchanged.
Direct users of the old method ID must change it to `sp_vla`. There is one owned
implementation, not two independently composable methods.
Primary category remains model_routing; cross-reference survey 5.1 for token pruning.

The full preset additionally reads the released last-block SigLIP Q/K, selects by
cumulative attention mass and protects Canny-edge patches in each camera. DINO
and SigLIP share selected patch indices; the LLM retains original positions. The
native vision output is second-to-last-block features; the last-block probe follows
the author's executable scorer, not a guessed replacement. All probe/edge costs
are timed. The author's code REDUCES attention mass as absolute Z increases,
contrary to comments in that repository. We preserve the code formula and label
it in runtime metadata. Cold/low-Z queries retain every token; we do not reverse
the rule to manufacture a speedup. One fixed spatial preset is used across suites.

The following describes the retained scheduling-only ablation:

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

## CogACT binding

The CogACT preset retains both scheduling and pruning with the same equations
and untuned parameters. It reads one native SigLIP view, shares its patch mask
across fused DINO/SigLIP features, and keeps original causal LLM positions and
cognition EOS. Native timm already executes its last block while returning the
second-last features, so reversible Q/K hooks do not replace the vision encoder.
All scoring and Canny work remains inside policy timing.

The Z gate uses the last **executed action's** vertical component, not observed
EEF height or an unexecuted prediction. No additional proprioception input is
introduced. CogACT predicts sixteen actions but normally executes eight; the
scheduling gate runs only when that executed prefix ends and emits one fitted
action at a time. DDIM/CFG and the diffusion head are unchanged. Scheduling alone
is an explicit ablation, not an additional algorithm or full-method claim.

Use `configs/experiments/cogact/sp_vla.yaml` with native LIBERO inputs; patch scoring reuses the native spatial resize to 224px.
Cold/low-Z queries retain all 256 patches even when this reduces the measured
speed benefit. Do not tune that gate using smoke outcomes.
