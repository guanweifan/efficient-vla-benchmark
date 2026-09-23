# DivPrune

Independent paper-based implementation of [DivPrune](https://arxiv.org/abs/2503.02175).
Reviewed [reference revision](https://github.com/vbdi/divprune/tree/799e2d950aa01ba7860907f5a6d86061f885dca6)
`799e2d950aa01ba7860907f5a6d86061f885dca6`. The reference repository is
CC-BY-NC-4.0; its code is not vendored or relicensed here.

Select projected visual embeddings before LLM layer zero. Using cosine distance,
seed with the patch whose nearest distinct neighbour is farthest away, then
repeatedly select the patch maximizing its minimum distance to selected patches.
Keep `round(N * keep_ratio)` unique positions, then restore positional order.
Distances use FP32, with deterministic first-position ties. Already-selected
positions are excluded explicitly; this avoids duplicate IDs in degenerate
identical-feature inputs where the released loop can select the same ID again.

OFT has one global budget across both cameras. Native nonvisual/action slots and
RoPE positions are unchanged. This is not VLA-Pruner with one score disabled.
No training, calibration or attention probes are used. Selection work is timed.
Full retention installs no hooks; reset and detach preserve native behavior.
