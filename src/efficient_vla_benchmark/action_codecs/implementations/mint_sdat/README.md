# MINT SDAT

Uses [released LIBERO-130 tokenizer weights](https://huggingface.co/huangrm/MINT-tokenizer-libero-130/tree/11cbedb05fee2f73b14325fe127bf128ab2e7805)
and [official SDAT source](https://github.com/RenMing-Huang/MINT/tree/bed9f0c0e2445a123814e89e4ba8c35a9578ce52/SDAT).
Native horizon16; scale lengths1/2/4 yield seven 512-entry IDs per block.
The codebook is shared across scales; learned residual convolutions and patchwise
position/rotation/gripper branches remain intact. Decode uses only indices,
residual accumulation, the released decoder and native gripper-logit argmax.

Load trainer.vae_wo_ddp strictly, matching the author's policy loader rather than
selecting EMA by reconstruction score. The actual checkpoint records native LIBERO
HDF5 actions and action_scale=1.0; do not silently substitute the normalization
defaults of a later LeRobot training recipe. DCT is an auxiliary training loss,
NOT a frequency-domain encode/decode transform in this released runtime.
Bypass the training builder that globally replaces Torch parameter initializers;
direct construction plus strict checkpoint loading preserves the same equations.

Use the separately frozen 16-frame matched group. Do not pad a 40-frame window or
mix its results into the main 40-frame ranking. No policy or closed-loop claim.
