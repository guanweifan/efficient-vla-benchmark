# FastV

Training-free transfer of [FastV](https://arxiv.org/abs/2403.06764).
Reference: [released HF implementation](https://github.com/pkunlp-icler/FastV/blob/d1659729b5bf1be225e99ee15783deeea80f63b1/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py),
revision `d1659729b5bf1be225e99ee15783deeea80f63b1`.

This is an independent implementation of the released last-query variant, not
the all-query averaging formulation in the early paper. At zero-based decoder
layer `pruning_layer - 1`, average the final query's attention across heads,
retain the top `round(N * keep_ratio)` visual keys, and prune before the next
layer. Stable positional order breaks score ties. Retain every nonvisual token.

OFT uses its native final EOS query and a global budget across both cameras.
No action-specific saliency claim is made. Original RoPE positions, native
bidirectional visibility and the native offset action readout remain unchanged.
The selected-row probe is included in timing; no full attention matrix is requested.
Full retention/disabled mode installs no hooks. Close is reversible and idempotent.

Validation covers an independent selection reference, protected tokens,
invalid configs, real-model no-op parity, actual shortened decoder rows,
reset determinism and native output after detach. GPU checks use caller-supplied
model/trace paths; shared LIBERO smoke is not formal accuracy.

## CogACT binding

CogACT uses its native final cognition EOS as the last query, with a 256-patch
single-camera pool. Causal visibility and original RoPE positions are retained;
the native diffusion head, sampler and RNG consumption are unchanged.
