# MiniVLA RVQ

Uses the [MiniVLA released LIBERO-90 tokenizer](https://huggingface.co/Stanford-ILIAD/pretrain_vq/tree/30ef2227f97dde1abb6d522ea80af85384235008):
eight-frame seven-dimensional blocks, seven residual 128-entry codebooks and
512-dimensional latents. Frozen MLPs/codebooks are loaded strictly, without the
training optimizer. Encoding uses the released Euclidean residual quantization;
decoding sums only codebook entries and applies the released decoder.

Reference: [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini/tree/0822b36227b5a771be4eb2680e34c559734c8fdc)
and its [VQ-BeT dependency](https://github.com/jayLEE0301/vq_bet_official/tree/09d4851288ca5deaaa1ab367a208e520f8ee9a84).
This measures the RVQ component, not full VQ-BeT or learned policy offset heads.
Target-suite q01/q99 normalization is frozen from calibration episodes; gripper
uses the native open=1/close=0 convention. No padding, codebook fitting or training.
