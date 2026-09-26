# EfficientVLA independent cores

Independent paper-based implementation of [section 3.3](https://arxiv.org/html/2506.10100v1).
The OFT ID `efficient_vla` implements token pruning only. The separately selected
CogACT ID `efficient_vla_diffusion_cache` implements section 3.4 only. Neither is
the full pipeline: no layer pruning or joint token/cache configuration is included.
The [author repository](https://github.com/YantaiYang-05/EfficientVLA) remains
unavailable through live access at the 2026-09-22 audit; neither core claims
author-code parity.

## OFT token-pruning core

Select a key set by task attention, add a fraction of remaining tokens by task
relevance, and fill the budget using maximum cosine distance from the ORIGINAL
key set (Eq. 4), not iterative farthest-point sampling. The two camera views get
separate budgets. Tokens retain original order, RoPE positions and action readout.

Task relevance follows Eq. 2: each visual QUERY's attention to contextual TEXT
KEYS is summed after head averaging. OFT's native bidirectional mask permits
this directly. The earlier implementation reversed the direction based on the
original CogACT's causal layout; that justification does not apply to OFT, and
historical results from that implementation are superseded, not relabeled.
Min-max normalization is omitted because only score rankings are used; it is
monotonic and cannot change selected tokens (stable original-order ties).

OFT conventions, not verified author defaults: post-block features and layer 1
(zero-based); 128 tokens and 8 key
anchors per view, task fraction 0.5. All are explicit configuration. No extra
full forward, finetuning or change of checkpoint. Full retention/disabled mode
does not attach a decoder adapter. Measure attention-probe and selection overhead.

Use `configs/experiments/openvla_oft/efficient_vla.yaml`. A migration experiment
tests this core on OFT; neither its latency nor smoke outcomes are the paper's
published CogACT results. Original complete-pipeline results must remain separate.

## CogACT diffusion-cache core

Use `configs/experiments/cogact/efficient_vla_diffusion_cache.yaml`. Equations 6–11
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

## Bounded development diagnostics

CogACT Dev-200 presets `efficient_vla_cache2` and `efficient_vla_cache3`
change the per-query refresh interval from 5 to 2 or 3 denoiser calls. They
retain separate attention/MLP caches, live residuals and conditioning, CFG
branches, native DDIM steps and native action RNG. No token pruning is combined.

The report selection additionally tests intermediate refresh intervals, without
changing the ten native denoising calls. Equal aggregate success counts do not
imply identical episode outcomes. Consult the central selection manifest and
ledger for current choices, paired timing and retained accuracy-boundary evidence.
