# OFT decoder extensions

`DecoderAdapter` changes one model instance, not a Transformers class or package.
`close()` removes its hooks and restores the original attention modules. Only one
strategy may be installed; batch-one, eval-mode, bidirectional OFT SDPA is the
supported execution path.

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

`IndexedKVStore` overwrites active positions rather than appending another frame.
The method owns cache lifetime and must reset when the episode or layout changes.
Native unused autoregressive cache outputs are disabled on the instrumented path;
the reference baseline is left untouched.

Tests cover dense no-op equivalence, key padding, original-position preservation,
protected tokens, identical-frame indexed reuse and reversible instance isolation.
The real-model GPU test compares three observations at absolute tolerance 1e-6.
An instrumented no-op is a diagnostic, never a replacement performance baseline.
