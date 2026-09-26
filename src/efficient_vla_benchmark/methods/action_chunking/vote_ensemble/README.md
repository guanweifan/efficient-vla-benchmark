# VOTE H8 overlapping-chunk ensemble

Independent adaptation of [VOTE v4, section 4.3, equations 8-10](https://arxiv.org/html/2507.05116v4#S4.SS3).
The [public source audited](https://github.com/LukeLIN-web/vote/tree/e1056932f52880ca58481673e4a1c74dc7d9f631)
did not expose an executable ensemble reference. This port is equation-based.

## Scope and cadence

Only CogACT's fixed joint checkpoint is registered: sixteen predicted actions,
eight executed actions, unchanged native DDIM/CFG and one forward every eight
controls. Store original chunks, indexed by their actual starting control step.
For every executed timestep, gather only predictions that still cover that SAME
timestep. Expire chunks at their end, not after a fixed number of control steps.
History belongs to one policy and clears on episode/instruction reset.

A chunk predicted at step 0 covers 0-15; the next, at 8, covers 8-23.
After the first chunk, each executed action normally has TWO votes. Compare both
with the newest available prediction using cosine similarity and strict threshold
0.5, split into similar/not-similar groups, and average the larger group.
Following equation 8, ties select the not-similar group: two agreeing votes average,
whereas two disagreeing votes select the older prediction. This two-vote
degeneracy is explicit; it is not the paper's dense adjacent-time multi-vote
committee and does not establish equivalent outlier rejection or published scores.
A single initial vote uses the current prediction. Votes never use already executed
actions, duplicate candidates, or previously ensembled outputs as new predictions.

One historical chunk is the only effective history budget for native 16-predict/
8-execute geometry. Unsupported history/cadence fails before model loading.
Both-zero cosine is 1; one-zero cosine is 0. The action codec retains native
Cartesian/axis-angle units and maps the already-discrete gripper to 0-1 for voting,
then back to LIBERO once. Native action noise/RNG consumption is unchanged.

## Exclusions and unsupported backend

No single-ACT token architecture, new action head, training, extra model calls,
extrapolated future actions or whole-paper speedup claim. Voting is an aggregation
contribution; fewer model calls or a speedup over native H8 is NOT assumed.

OpenVLA-OFT's current checkpoint predicts eight and executes eight: consecutive
chunks do not overlap. A one-vote no-op is not a reproduction. It is unregistered
rather than switched to H1 or given hidden auxiliary inference. Longer trained
chunks or additional inference require a separately reviewed scope.

Use `configs/experiments/cogact/vote_ensemble.yaml` with the unchanged common
H8 timing replay and Dev-200. Historical H1 results are retired from the primary
comparison, never relabeled or numerically compared as an H8 improvement.

## Verification

CPU tests cover physical-time alignment throughout a chunk, exact two-vote/tie
behavior, independent prediction storage, expiry, bounded history, reset and
unsupported pairs. The real-model test compares the H8 controller to independently
aligned native sixteen-step chunks, verifies query counts, disabled/reset/detach
parity and unchanged diffusion RNG. Measurements belong in the central ledger;
passing mechanism tests alone is not a success-rate or acceleration claim.
