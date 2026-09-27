# X-Tokenizer: conditional command-space reconstruction

[Released weights](https://huggingface.co/x-square-robot/X-Tokenizer) revision
`bb5a47a36079d085d55c303fdf73223bd3dedd1d`; [runtime](https://github.com/X-Square-Robot/X-Tokenizer)
revision `9733990b16a07d51fb8b5d305ad8c16abb57aef2`. Preserve all four 2048-code
RVQ stages, temporal compression four, the native 26D layout and **Franka ID2**.
None/default would select X2Arm and is not acceptable. Only the left-arm ten
canonical dimensions are present; mask all missing DOFs using the native API.

LIBERO stores step-relative OSC commands, whereas this release expects absolute
poses. Use an explicitly scoped, invertible **command-integrated trajectory**:
start at the initial observed EEF pose, accumulate 0.05-m-scaled position deltas
and left-compose 0.5-rad-scaled axis-angle increments. Native Panda finger ranges
give observed opening `(left-right)/0.08`; command opening is `(1-gripper)/2`.
Rotations use the first two matrix rows. Inverse differencing returns raw commands.
The mapping is a lossless coordinate embedding, NOT a prediction of physical robot
motion or equivalence to recorded future poses. Its cost is inside encode/decode.

Only the initial observed eight-value state is decoder side information; never use
future observed states or targets. This is a **conditional** rate–distortion result:
the common observation is excluded from token payload bits and must be disclosed
when comparing unconditional codecs. No policy head or success claim is implied.

Fit native TDigest statistics only on calibration command trajectories, using the
released training-statistics horizon 32 and frozen random seed. Forty-frame inference
uses the supported dynamic-length API. Neither weights, RVQ nor stats change during
evaluation. Preserve clipping/rotation/conditional reconstruction errors; never fill
malformed rotations with identity. The representation adaptation and unknown original
pretraining overlap limit claims to this disclosed reconstruction workload.
