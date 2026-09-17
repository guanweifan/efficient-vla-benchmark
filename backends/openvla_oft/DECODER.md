# OFT decoder extensions

`DecoderAdapter` changes one model instance, not a Transformers class or package.
`close()` removes its hooks and restores the original attention modules. Only one
strategy may be installed; batch-one, eval-mode, bidirectional OFT is supported.
Native/no-op execution retains SDPA. A strategy can explicitly select `eager`
when its algorithm consumes complete attention probabilities.

`TokenLayout` owns the two-camera visual range, proprioception, variable-length
language, action placeholders and the checkpoint's original action-head readout.
Strategies return masks in current-row coordinates. Non-visual tokens are
protected, original positions drive RoPE, and the final hidden tensor is scattered
back to canonical positions before the native head reads it. Removed visual rows
are zero placeholders at this final boundary, not extra decoder computation.

Methods can request selected attention rows without forcing every decoder layer
to materialize a full attention matrix. Softmax still normalizes over every key,
and the head-reduced probabilities are returned in original-key order. Additional
probe and selection work is part of measured method latency.

For eager execution, QK/softmax is evaluated once and reused for both P @ V and
the requested statistics, following standard LlamaAttention. It does not also
call SDPA. Full-row probes can request the state's positions tensor directly,
avoiding redundant search/gather/host synchronization. Other requests remain
validated, including out-of-range requests. Padding and original key positions
are respected in both paths. Eager versus fused SDPA can differ numerically in
low precision; eager is tested against an independent standard eager reference,
not asserted to be bitwise identical to fused attention.

`IndexedKVStore` overwrites active positions rather than appending another frame.
The method owns cache lifetime and must reset when the episode or layout changes.
Native unused autoregressive cache outputs are disabled on the instrumented path;
the reference baseline is left untouched.

Tests cover dense no-op equivalence, key padding, original-position preservation,
protected tokens, identical-frame indexed reuse and reversible instance isolation.
The real-model GPU test compares three observations at absolute tolerance 1e-6.
An instrumented no-op is a diagnostic, never a replacement performance baseline.
