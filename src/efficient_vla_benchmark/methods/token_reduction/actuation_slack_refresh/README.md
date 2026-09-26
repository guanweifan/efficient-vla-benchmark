# Actuation-Slack Refresh: independent deletion and KV-reuse instruments

Implements the independently selectable deletion and KV-reuse instruments of
[Actuation-Slack Refresh](https://arxiv.org/abs/2608.00391), not a composition with
VLA-Cache or VLA-Pruner. IDs: `actuation_slack_refresh_deletion` and
`actuation_slack_refresh_kv_reuse`. Each run selects exactly one instrument.

The first served chunk is dense. Every served chunk unconditionally schedules
one dense pass on the **same prepared frame**, whose action output is discarded.
Only this complete dense pass can update the next chunk's gate; sparse served
attention is never harvested. Subsequent chunks retain the top-ranked primary-view
patches under that previous clean gate. OFT wrist tokens are never pruned.
A dense first chunk also receives a refresh, preserving the unconditional schedule.

The gate is mean action-query attention at explicitly configured layers (default
zero-based layer 15). The publication's main contribution varies gate families;
the layer choice is a declared port convention, not verified author code. Ratio
0.1 follows the paper's 90% skipping instrument. Native masks, original positions
and action readout remain unchanged. No training or oracle actions are used.

The shared backend worker executes refresh asynchronously on a separate CUDA stream
while queued actions may execute. It always joins before the next model query,
reset or detach; failures propagate. Same-frame preparation is shared read-only,
not repeated, and no proprioception is normalized twice. RNG is restored after
the discarded pass so a stochastic action head cannot alter later served noise.

**Accounting:** the common monitor joins the worker before stopping the timer.
Thus the main latency/throughput row includes primary **plus all auxiliary work**.
Separate primary-forward and refresh diagnostics do not imply a reduction in total
computation. Primary-forward timing excludes common observation preparation and
must not be relabeled end-to-end serve latency. A configured 400 ms actuation budget
is only a nominal check; unpaced simulation/replay does not establish real-robot
overlap, energy savings, or hardware deadline guarantees.

## CogACT binding

CogACT uses native final cognition-to-image attention as the disclosed gate
adaptation. It does not invent action-to-patch attention inside DiT. The core
claim retained here is gate provenance plus unconditional same-frame dense refresh;
subsequent served chunks retain 26/256 patches by default. The auxiliary call runs
the complete native stochastic action head but discards its actions and restores
RNG, so the next served sample consumes exactly the native noise sequence.

## KV-reuse instrument

The reuse component implements equations 1-2 with the same pure-attention clean
gate as deletion. All visual keys remain visible with their native causal or
bidirectional constraints. Selected primary-view queries and MLP rows execute;
skipped rows use one-chunk-old per-layer KV from the complete dense refresh.
The first served chunk is dense. Wrist/nonvisual rows are always computed.

The dense KV base is immutable during serving: active-row splices are temporary,
never recursively published. Every auxiliary pass replaces all layer KV and the
gate, even after a dense first chunk. Gate probing, cache storage/splicing and the
complete discarded action head are included in the common timing boundary.
This component does not implement the VLA-Cache static/entropy gate or claim its
composition results. Configuration layer choices and CogACT cognition-attention
adaptation have the same explicit limitations as the deletion instrument.

Tests check immutable clean KV, missing-base rejection and real-checkpoint
cold/no-op/reset/detach/RNG behavior. Measurements must additionally pass the common
smoke, Dev-200 and full-cost replay before this scope enters the primary ledger.

## Bounded development diagnostics

CogACT Dev-200 presets `actuation_slack_refresh_keep25` and
`actuation_slack_refresh_keep50` raise the primary-view retention ratio from 0.1
to 0.25 or 0.5. Dense first chunks, unconditional same-frame dense refresh,
gate provenance and RNG restoration are unchanged. Timing must include refresh;
a better success rate cannot justify omitting this cost.

Fixed CogACT Dev-200 scored 137/200 at retention 0.1, 178/200 at 0.25 and
182/200 at 0.5, versus native 183/200. This isolates strong retention sensitivity
while keeping the refresh mechanism. A near-native success rate does not remove
the extra dense-pass cost or establish real-robot serve-latency acceleration.
