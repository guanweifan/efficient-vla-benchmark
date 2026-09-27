# Pinned external code

- `benchmarks/libero`: official simulator tasks and resources.
- `runtimes/transformers_openvla_oft`: the native OFT Transformers fork.
- `runtimes/dlimp`: OFT training-data dependencies.
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
The root `NOTICE` preserves terms and attribution for the new source-based token
ports. TEAM-VLA and VLA-IAP have paper-only implementations with no runtime source
dependency; their READMEs identify the missing author-code verification.

Optional standalone action codecs use pinned release assets and source subsets,
never the policy application's dependency stack. Run
`python -m efficient_vla_benchmark.action_codecs.fetch_sources --output
third_party/runtimes/action_codecs` to fetch VQ-VLA/X-Tokenizer source and licenses
without a separate checkout. This generated directory stays ignored. Every file's
upstream and locally adapted checksum is recorded; VQ-VLA changes only package
import prefixes. Pass the selected source directory via the codec runner's
`--sources` argument. Released weights/processors remain external CLI resources.
