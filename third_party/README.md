# Pinned external code

- `benchmarks/libero`: official simulator tasks and resources.
- `runtimes/transformers_openvla_oft`: the native OFT Transformers fork.
- `runtimes/dlimp`: OFT training-data dependencies.
- `runtimes/prismatic_cogact`: CogACT's pinned Prismatic/OpenVLA classes.
- `runtimes/llm_awq`: AWQ source primitives and packed-kernel source.
- `methods/specprune_vla`: the paper reference implementation.
- `methods/vla_cache`, `methods/vla_cache_transformers`: VLA-Cache reference sources.
- `methods/adp`, `methods/vla_pruner`, `methods/safe_pruner`: pinned audit/test
  references; their application stacks are not imported into the active backend.

Actual revisions are the parent repository's gitlinks, not floating branches.
Method READMEs distinguish runtime imports from audit-only reference code. Do not
install reference application stacks when their dependencies conflict with the
selected backend. Retain upstream licenses and copyright notices. First-party
adaptations belong in methods/, never as unexplained changes to reference sources.
The root `NOTICE` preserves terms and attribution for source-based ports.

The fixed commits of all fourteen submodules were verified publicly readable on
2026-10-21. `dlimp` and `methods/specprune_vla` have no explicit license file at
those revisions; applicable permissions remain to be confirmed. Source availability
is not a license grant. See the unresolved entries in [NOTICE](../NOTICE).

Optional standalone action codecs use pinned release assets and source subsets,
never the policy application's dependency stack. Run
`python -m efficient_vla_benchmark.action_codecs.fetch_sources --output
third_party/runtimes/action_codecs` to fetch VQ-VLA/X-Tokenizer source and licenses
without a separate checkout. The same fetcher includes MINT, SpatialVLA and the
MiniVLA RVQ reference dependency. SpatialVLA declares MIT in its pinned README
but that revision has no LICENSE file; preserve the declaration and do not invent
missing license text. This generated directory stays ignored. Every file's
upstream and locally adapted checksum is recorded; VQ-VLA changes only package
import prefixes. Pass the selected source directory via the codec runner's
`--sources` argument. Released weights/processors remain external CLI resources.
