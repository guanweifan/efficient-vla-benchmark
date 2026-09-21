# EfficientVLA token-pruning core

Independent paper-based implementation of [section 3.3](https://arxiv.org/html/2506.10100v1).
This is NOT the full EfficientVLA pipeline: layer pruning and DiT caching are absent.
The paper-linked author repository could not be accessed during this integration.

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
