# AAC

[Adaptive Action Chunking](https://arxiv.org/html/2604.04161v2), equations 2--10:
draw independent action chunks with shared cognition, estimate per-step Gaussian
translation/rotation entropy and Bernoulli gripper entropy, find the largest
increment in prefix-average entropy, and enforce a cumulative movement lower
bound. The selected prefix is really executed by the common controller.

The [author LIBERO selector](https://github.com/Adaptive-Action-Chunking/libero/tree/ff8c8e421f7805b5845a72d07fbd8f1171df74bf)
pins candidate zero, a two-command minimum, covariance ridge 1e-6, and movement
threshold 3. Its gripper movement weight is 0.2, whereas the paper appendix writes
1; this port follows the released code and exposes that weight explicitly.
Rotation increments are composed as ordered quaternions, not summed as vectors
or replaced with a principal-angle library convention. See the root NOTICE.

The [author server](https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample/tree/11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a)
exports raw normalized predictions for entropy, plus denormalized commands for
movement. We retain that division using native CogACT normalization, clipping,
binary gripper and axis-angle delta semantics. The original candidate-zero tensor
is returned to the unchanged native postprocessor. No candidate averaging, extra
VLM passes or sampling-step reduction is introduced.

Default K=20 follows the paper; K=5 and K=10 expose its tested speedward sample
budgets. K<4 is rejected because a three-dimensional empirical covariance needs
four samples to have full rank; a one-sample no-op is not a reproduced entropy
method. Native DDIM, CFG pairing and sixteen predictions remain unchanged.
Additional candidates consume real independent noise. The original H8 baseline
is the comparator; the execution prefix is adaptive, not a relabeled fixed H8.

Only CogACT is supported. OFT's deterministic action head does not supply the
required stochastic candidate distribution. This is an inference-only core port,
not reproduction of the authors' GR00T training or their optional backward
candidate selector. All candidate generation, CPU transfer, entropy, ordered
rotation composition and control costs belong in the common timing boundary.
Higher success or acceleration is not assumed.

Reset clears every query/horizon; hooks are per-instance and reversible. Disabled
mode attaches no sampler wrapper. Unsupported geometry and incomplete selection
fail explicitly. Model/checkpoint paths remain caller-provided CLI resources.
