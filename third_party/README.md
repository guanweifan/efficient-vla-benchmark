# Pinned external code

- `benchmarks/libero`: official simulator tasks and resources.
- `runtimes/transformers_openvla_oft`: the native OFT Transformers fork.
- `runtimes/dlimp`: OFT training-data dependencies.
- `runtimes/llm_awq`: AWQ source primitives and packed-kernel source.
- `methods/specprune_vla`: the paper reference implementation.
- `methods/vla_cache`, `methods/vla_cache_transformers`: VLA-Cache reference sources.

Actual revisions are the parent repository's gitlinks, not floating branches.
Method READMEs distinguish runtime imports from audit-only reference code. Do not
install reference application stacks when their dependencies conflict with the
selected backend. Retain upstream licenses and copyright notices. First-party
adaptations belong in methods/, never as unexplained changes to reference sources.
