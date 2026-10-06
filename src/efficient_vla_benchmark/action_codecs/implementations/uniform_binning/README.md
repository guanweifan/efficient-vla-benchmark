# Uniform binning baseline

Uses OpenVLA's 256 equally spaced boundaries and 255 midpoint decoder,
including the clipped upper-end index. LLM-vocabulary offsets are removed
bijectively; report seven scalar tokens and 56 nominal bits per frame.
Calibration q01/q99 bounds use training-only corpus episodes; gripper follows
the native open=1/close=0 convention before inversion to raw LIBERO commands.
This is a classical baseline, not an additional paper or a policy replacement.
