# VOTE trajectory ensemble

Category: action_chunking (survey §5.5). Inference-only on native OpenVLA-OFT.

Implement the independent trajectory ensemble contribution from
[VOTE v4, §4.3, equations 8–10](https://arxiv.org/html/2507.05116v4#S4.SS3).
The single-ACT token architecture, new trained action head and whole-paper speedup
claims are excluded. This component is an action-selection method, not intrinsically
a model-latency accelerator.

[Public repository audited](https://github.com/LukeLIN-web/vote/tree/e1056932f52880ca58481673e4a1c74dc7d9f631):
the actual ensemble function was not located. This is our independent equation-based
implementation, not a claim of executing the authors' voting code. No upstream
runtime, training, checkpoints or code vendoring is necessary.

For each physical timestep, gather predictions for that SAME timestep from current
and previous chunks. Compare their cosine similarity with the newest prediction,
divide into similar (`> threshold`) and not-similar groups, and average the larger
group. Ties select not-similar. Default threshold is 0.5, with three past timesteps
plus the current prediction (the paper's illustrative K=3). This is an untuned preset.
Two zero vectors have similarity 1; one zero vector has similarity 0.

The OFT binding maps environment gripper commands back to the paper's 0–1 convention
before cosine/voting, then thresholds the voted gripper once on return. OFT already
discretizes its native gripper; that is the input to this port, not hidden confidence
logits. XYZ/rotation remain in native denormalized action units. The NumPy voting
algorithm and action-space codec are separate for future model bindings.

Full eight-action predictions are retained and indexed by their starting control
step; only one action is executed before querying again. Presets require horizon 1
and reject horizon 8, which would eliminate meaningful overlapping predictions.
`baseline_horizon1.yaml` is the matched-cadence speed reference. Also retain native
horizon-8 baseline as a deployment reference: horizon-1 voting incurs many more
model calls than horizon-8 open-loop execution. Never attribute that difference to
the voting kernel or claim the single-ACT paper's speedups here.

Use `vote_ensemble.yaml` through the shared evaluate/benchmark commands. Control
replay is mandatory; all history/codec/voting overhead is timed. `enabled: false`
is a native horizon-1 no-op. Reset and instruction changes clear all predictions.
Success-rate/trajectory benefits remain unmeasured until a scheduled formal run.

## CogACT binding

The independent ensemble equations and LIBERO action codec are shared without
importing the OFT runtime. Native CogACT still predicts sixteen actions using
the same DDIM/CFG sampler. The paper's committee spans adjacent observation times;
the default K=3 cannot be used with H=8, even though two sixteen-step chunks would
geometrically overlap. H=1 preserves all adjacent-time candidates; a dedicated
native H=1 reference isolates voting overhead. Retain native H=8 as a deployment
reference, not a denominator interchangeable with H=1.

Use `configs/experiments/cogact/vote_ensemble.yaml`. History is limited to at most
fifteen earlier control times for this model. No native adaptive ensemble is
enabled. Raw gripper values are already discretized by native prediction; the
shared codec maps signed LIBERO commands to 0-1 before voting and back once.
Tests verify the committee against independent real-chunk calculations, no-op
parity, reset and unchanged diffusion RNG consumption. This is the independent
inference voting contribution, not VOTE's trained single-ACT architecture.
