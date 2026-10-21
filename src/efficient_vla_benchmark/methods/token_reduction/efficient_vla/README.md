# EfficientVLA independent cores

Independent paper-based cores from [sections 3.2-3.4](https://arxiv.org/html/2506.10100v1).
The OFT ID `efficient_vla` implements token pruning only. The separately selected
CogACT ID `efficient_vla_diffusion_cache` implements section 3.4 only. The
OFT/CogACT ID `efficient_vla_layer_pruning` implements section 3.2 only. None is
the full pipeline: these independent cores are never jointly selected.
The [author repository](https://github.com/YantaiYang-05/EfficientVLA) remains
unavailable through live access at the 2026-09-22 audit; none of these cores claims
author-code parity.

## OFT token-pruning core

Select a key set by task attention, add a fraction of remaining tokens by task
relevance, and fill the budget using maximum cosine distance from the ORIGINAL
key set (Eq. 4), not iterative farthest-point sampling. The two camera views get
separate budgets. Tokens retain original order, RoPE positions and action readout.

Task relevance follows Eq. 2: each visual QUERY's attention to contextual TEXT
KEYS is summed after head averaging. OFT's native bidirectional mask permits
this directly; the reverse direction is not equivalent.
Min-max normalization is omitted because only score rankings are used; it is
monotonic and cannot change selected tokens (stable original-order ties).

OFT conventions, not verified author defaults: post-block features and layer 1
(zero-based); 128 tokens and 8 key
anchors per view, task fraction 0.5. All are explicit configuration. No extra
full forward, finetuning or change of checkpoint. Full retention/disabled mode
does not attach a decoder adapter. Measure attention-probe and selection overhead.

Use `configs/experiments/openvla_oft/efficient_vla_token.yaml`. A migration experiment
tests this core on OFT; neither its latency nor smoke outcomes are the paper's
published CogACT results. Original complete-pipeline results must remain separate.

## CogACT diffusion-cache core

Use `configs/experiments/cogact/efficient_vla_cache.yaml`. Equations 6–11
cache each DiT block's self-attention and MLP outputs **separately**. At the first
denoiser call and every N calls afterwards, both outputs are recomputed; intervening
calls reuse them while retaining live residual inputs, native normalization,
conditioning, positional embeddings and final action projection. Default N=5
gives two feature-refresh steps during ten native DDIM calls, NOT two DDIM steps.

The periodic counter is the sampling-call index, not the original training-time
timestep integer: native respacing passes e.g. 90, 80, ...; taking those values
modulo five would refresh every call and contradict the paper's stated N-step
schedule. This indexing convention is explicit. The default interval follows
the paper's two-refresh/ten-step configuration, without tuning smoke outcomes.

Instance-scoped forward wrappers cover native `forward_with_cfg`, which calls
`forward` directly. Cache entries retain both conditional/unconditional batch
rows without mixing; native CFG 1.5 and DDIM noise/RNG remain unchanged. Caches
clear before and after every policy query, on reset and on denoiser errors.
Disabled or N=1 leaves native forwards untouched; close restores instance
attributes. Feature geometry changes inside a query fail explicitly.

Independent native-DiT tests compare the complete cached residual recurrence
and CFG outputs, count both module types, and test reset/errors/no-op/detach.
Real-model tests additionally verify all twelve blocks, ten sampler calls,
two refreshes and eight reuses, native parity and deterministic replay.
The common control timing includes wrapper, lookup and bookkeeping overhead.

## Deferred CogACT token-pruning scope

Equation 2 requires visual-query to text-key attention, but the native causal
visual prefix cannot attend to following instruction tokens. Reversing Q/K,
removing its causal mask, or substituting the final cognition score would change
this published rule. The paper reports CogACT, so this is an unresolved
paper/native-implementation discrepancy, not evidence that the authors' method
cannot work. Keep this scope unregistered until a verifiable attention convention
resolves it; the cache-core result cannot stand in for token pruning.

## Static layer-pruning core

Equation 1 scores every native decoder layer as one minus its input/output cosine
similarity, averaged over ALL token positions within each sample, then equally
across samples. Calibrate on a fixed balanced RLDS training subset with the common
`calibrate` command; never use Dev/final episodes. The default 32 samples/seed zero
are disclosed benchmark calibration choices, not a verified author recipe.

Select the lowest-scoring `prune_layers` indices with stable index tie-breaking,
remove those non-contiguous layers and preserve the order of survivors. The decoder
module list is physically shortened; removed weights move to CPU for reversible
detach, with cache layer indices adjusted to the compact stack. They are not left
resident on GPU behind no-op forward hooks. Detach restores the exact original
modules, indices, depth and device without changing weights. Native attention,
all visual tokens, final normalization, action head and H8 cadence remain unchanged.

This is static model compression (survey Section 3.3), also relevant as a depth-
reduction reference near Section 5.3; it is not adaptive layer routing or a new
paper. PruneNet MLP compression is excluded, as are token pruning and DiT caching.
Calibration scores and their training/checkpoint/source identities live in a
separate checksummed-provenance artifact; changing the number of removed layers
does not require recalibration. Every selected count gets its own Dev/timing pair.

## Selected configurations

The three independent scopes use `efficient_vla_token.yaml`,
`efficient_vla_layer.yaml` and `efficient_vla_cache.yaml` on their supported
backends. Cache intervals count feature refreshes, not fewer DDIM calls. Final
settings and accuracy-boundary evidence are in the central selection manifest.
