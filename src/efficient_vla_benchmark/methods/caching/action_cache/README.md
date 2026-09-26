# ActionCache

Independent [paper section 3](https://arxiv.org/html/2607.06370v2#S3) port:
fixed sparse ternary projection of native VLM cognition, nearest cosine retrieval,
a strict hit threshold, actual cached intermediate noisy actions, and fresh-condition
refinement. Misses execute the complete native sampler. No replacement of a cached
noisy intermediate with a newly noised clean action is permitted.

CogACT uses DDIM epsilon predictions, not the paper's flow integrator. Store native
DDIM states at each schedule index and resume the matching native tail with current
cognition and CFG. The native scheduler, action coordinates and normalization remain
unchanged. Cached states retain one action batch, duplicated only for native CFG.
Preserve unused eta=0 scheduler RNG draws on skipped steps; do not shift subsequent
native query noise. Current VLM cognition is computed on every query.

Offline population uses independent native successful rollouts, with pending pairs
committed only after episode success. The frozen bank is shared unchanged by all
Dev/timing shards. Never update it using evaluation success labels. This matches
the paper's prefill/steady-state separation, not its optional online replacement.
The builder retains unsuccessful collection outcomes and checks initialization
states against all final-evaluation states. Prefill compute/storage is separate
from timed lookup/refinement; no extra training is needed.

Keys use the architecture's existing final 4096D cognition output; 500 projected
dimensions, density 0.01 and fixed seed by default. Each projection row has exactly
floor(D*p/2) positive and negative entries (at least one); it is data-independent.
Suite partitions respect CogACT's different action normalization statistics;
retrieval is across tasks within each partition, not across incompatible coordinates.
One parameter setting serves all suites. Bank checksums/checkpoint/sampler identity
are validated before use; data are read-only NPZ, never executable pickle.

The implemented scope excludes zero-refinement actions and online evaluation cache
population. Disabled mode uses untouched native sampling. Reset clears query state,
not the immutable offline bank; close reverses only this instance's sampler wrapper.

## Population and frozen artifact

Use a CogACT/LIBERO runtime with CUDA and EGL mapped to the same assigned GPU.
The population command delegates every rollout to the shared H8 episode runner;
it owns only independent-state selection and successful-pair capture:

```bash
python -m efficient_vla_benchmark.methods.caching.action_cache.implementations.cogact.populate \
  --model-path "$MODEL_DIR" --libero-root "$LIBERO_ROOT" \
  --output-dir "$POPULATION_SHARD" --shard 0 --shards 4
python -m efficient_vla_benchmark.methods.caching.action_cache.freeze \
  --shards "$SHARD_0" "$SHARD_1" "$SHARD_2" "$SHARD_3" \
  --libero-root "$LIBERO_ROOT" --output-dir "$BANK_DIR"
evb evaluate --config configs/experiments/cogact/action_cache.yaml \
  --model-path "$MODEL_DIR" --artifact-path "$BANK_DIR" --libero-root "$LIBERO_ROOT" \
  --episode-manifest "$DEV_MANIFEST" --output-dir "$RUN_DIR" --no-record-traces
```

Collect each shard exactly once. Across all shards this is exactly two native
episodes per task; an unsuccessful episode is not replaced. Freeze checks all
eighty outcomes, successful-pair accounting, actual saved initialization states
against all fifty standard states per task, bank geometry and checkpoint identity.
Existing outputs are rejected. Cache capacity is 3000; overflow is an error, not
silent test-guided eviction. Population cost excludes loading and is not hidden in
steady-state speedup.

The current frozen bank contains 1412 pairs from 77/80 successful independent
episodes. Initial threshold 0.90 was chosen before Dev evaluation from leave-one-
episode-out prefill similarity: expected hits span roughly 10–32% across suites,
whereas 0.95 nearly disables retrieval in some suites. This is a disclosed
CogACT-specific choice, not the original models' threshold.
