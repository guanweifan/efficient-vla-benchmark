# Falcon: partial denoising on native CogACT

Independent implementation of the full DDIM core in
[the ICML paper](https://proceedings.mlr.press/v267/chen25ao.html),
Sections 3.1--3.3 and Appendix B.1 Algorithm 2. The
[released code](https://github.com/chjchjchjchjchj/Falcon/tree/9a72d315dd072d46f3498bf6909784914c216e39)
is a convention cross-check, not a bundled dependency or a bitwise-parity claim.

## Core and explicit reference choices

The first query runs the unchanged full native DDIM chain and stores historical
partially denoised actions. Subsequent queries evaluate ALL bounded historical
candidates in one batched current-observation epsilon probe. Cumulative alpha
converts these into clean-action estimates. Compare their first overlapping
prefix against the immediately preceding prediction's unexecuted tail.

Use the released distance units (L2 divided by the number of scalar reference
entries, NOT RMSE), inclusive distance threshold, strict minimum-noise gate,
and softmax of NEGATIVE noise level over eligible candidates. This resolves the
paper's indicator-inside-exponent notation using its stated eligible-set
selection and executable helper. Empty eligibility or probabilistic exploration
returns to native Gaussian initialization. Candidate selection uses its own
seeded generator. An active candidate is a historical noisy trajectory, not
a clean action reused without verification or fresh observation conditioning.

Follow the paper's pre-update buffer insertion and update the reference after
EVERY generated chunk. The release stores post-update latents under the input
noise label, and its later path updates last_latent_traj rather than last_action.
Those inconsistent lines are not copied. Buffer eviction follows paper priority:
oldest query, then highest noise. The release uses distance from the noise floor.
All qualified entries are eligible; for buffer_size above 20, this differs from
the helper's extra top-20-distance cap. Default capacity 20 makes that cap moot.
These choices are explicit; this is not a reproduction of release bugs.

## Native contract and cost

CogACT's sixteen predictions/eight executed actions provide an eight-action
overlap. No new head, trajectory padding, shifted latent, training or action
cadence is introduced. The previous reference remains in native normalized
action coordinates, before clipping, unnormalization and gripper conversion.

Keep the original finite-noise start, native ten-step grid, DDIM update routine,
CFG and current cognition. Batched probes place ALL conditional candidates before
ALL unconditional candidates, retaining each candidate's matching timestep.
Partial sampling resumes at the stored latent's actual scheduler index and uses
the untouched native DDIM tail, including its clean endpoint.

Initial noise and the native ten unused eta-zero random draws remain aligned;
selection randomness is private and resets by episode seed. First-query and
forced Gaussian sampling are native-equivalent. State is instance-local,
bounded and cleared by reset/close. Interrupted/incorrect-cadence histories are
rejected rather than reused.

Full-cost control replay includes the VLM, batched probes, selection, latent
storage, RNG accounting, remaining sampling and output conversion. Report both
denoiser batch calls and candidate-equivalent evaluations; one batched probe is
not free and may outweigh skipped sampling. No published speedup is assumed.
Only native H8 control evaluation is supported for this history-dependent method.
OFT has no diffusion sampler or unexecuted H8 overlap and is not registered.

## Parameters and validation

Default distance_threshold=0.01, exploration_rate=0.1, temperature=1,
min_noise_level=10 (original training-noise index), buffer_size=20.
Increase the distance threshold or lower the noise floor to test speedward
headroom. Keep active current-observation verification and stochastic exploration;
never relabel a disabled/native configuration as an accelerated result.

Tests cover independent posterior/distance/probability equations, buffer noise
labels/eviction, CFG batch order, native first-query/Gaussian parity, actual call
accounting, RNG, reference refresh, reset/detach and wrong-cadence rejection.
Common four-suite smoke, Dev-200 selection and paired replay remain separate
measurement gates; registration alone is not a completed reproduction.
