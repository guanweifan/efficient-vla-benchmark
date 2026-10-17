# MotionVQ — scoped VQVLA algorithm component

[Paper](https://arxiv.org/abs/2607.24148v1);
[public OFT implementation](https://github.com/ACA-Lab/VQVLA/tree/bc8b69c4fd4fd24f1a3b4694d58f78caf6c6f79a).
This is the paper's independent motion-aware dual-codebook contribution on the
224 language decoder projections, not its complete algorithm/hardware co-design.

Preserved: high VQ[256,2,256], low VQ[128,2,64], Hessian-diagonal weighted
block-local k-means, vector length two, first-query high precision and subsequent
motion-dependent selection. Calibration follows the public exporter's 2D block
codebooks and final-assignment/update ordering. Native BF16 centroids are retained.
The paper's prose about per-column-group codebooks is not silently mixed with
the released block-level format. This port does not claim identical full-model packs.

The public OFT router uses >= at the threshold whereas the paper says >; this
official-code variant preserves >=. H8 reads the last actually executed command,
not an unexecuted future action. CogACT thus uses its eighth command rather than
the unused H16 tail. Reset clears state. One global threshold applies to all suites.

Unlike the released dense reconstruction/two-model execution, both compressed
banks remain resident, and a Triton kernel decodes 6/8-bit indices and gathers
centroids inside each GEMM tile. No full BF16 weight cache or hidden fallback.
Codebook bytes, both index banks and all routing/gather costs count toward runtime.
The kernel uses native floating matrix multiplication; it does not claim the
paper's spatial input aggregation or temporal cached-product reuse across GEMM
columns. Temporal reuse here is not cross-frame robot-action caching. Both are
defining multiplication-reduction mechanisms, not simply accelerator-specific
implementation details. This registered component remains diagnostic and excluded
from the full-VQVLA comparison; its speed cannot establish faithful full-method
GPU performance. The paper's own GPU slowdown does not validate this incomplete
component or prove a faithful GPU reuse implementation impossible.

Prefill-size-dependent launch tiles preserve the original K=32 reduction and
match the compact kernel exactly in GPU regression tests. Six-bit packed indices
need cross-byte gathers; fewer stored bits do not guarantee faster GPU execution.
Threshold 0.8 remains the public-code reference. The 1.0 candidate retains dynamic
switching while reducing low-bank use; both require their own Dev and timing data.

The public full-model exporter also quantizes vision, action, embeddings and
normalization tensors. Those are explicitly excluded here, following the paper's
stated transformer-backbone bottleneck scope. Training-only balanced RLDS captures
replace author per-suite calibration; no evaluation trajectories or new checkpoint
downloads are used. Only codebooks are fitted, not policy parameters.
