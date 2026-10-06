# SpatialVLA spatial action tokenizer

Runs the author's [SpatialActionTokenizer](https://github.com/SpatialVLA/SpatialVLA/blob/18fd74b2a633ec8d9ec7aadcd803969555cc9fbd/model/action_tokenizer.py)
with the released pretrained Gaussian bin grids from
[spatialvla-4b-224-pt](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt/tree/886c4557dfde53b44c0f74fde7d89bf83a26c3cd).
Only the small processor config is required, not policy weights.

Translation/rotation/gripper use disjoint alphabets of 4096/4096/2 entries.
Report three IDs/frame and 42 fixed-width bits/frame in the joint 8194-entry
alphabet, not an unannounced positional 12+12+1-bit coding scheme.
Normalize raw LIBERO coordinates with calibration-only q01/q99 and native open-gripper
convention. This is a disclosed transfer of spatial quantization to normalized OSC
coordinates, not a physical Euler-angle conversion or a test of learned embedding
adaptation. Do not attribute full SpatialVLA policy performance to this experiment.
