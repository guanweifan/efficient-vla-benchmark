# DP-Cache

Independent implementation of the stable-window denoiser-output broadcast contribution
in [section 5.2.2](https://arxiv.org/html/2604.24447v1#S5.SS2.SSS2).
No V-AEFusion, model compilation or combination with another method is included.
This is not a reproduction of the paper's model checkpoints or published scores.

The first call in a configured stable window computes and saves the native DiT
output. Subsequent calls reuse it until the refresh interval expires. Calls outside
the inclusive window always compute. CogACT still executes all native DDIM scheduler
transitions with current noisy actions; only denoiser evaluation is skipped.
Cache both CFG rows before native guidance, never a conditional-only proxy.
Reset at each action query/episode/error, and restore instance forwards on close.

The ten-call CogACT schedule uses zero-based call indices, NOT training timesteps.
The initial candidate window 2–7 and interval 2 are explicit migration parameters,
not a claim that the paper's 20–80 window applies unchanged. A native stability
probe on sixteen training observations supports a middle window: adjacent-output
relative L1 changes rise sharply at the final steps. The ignored training-only
profile is retained separately from Dev-200, which checks the reuse interval and
neighboring window boundaries. This is not a learned sampler or the paper's
original checkpoint schedule. Include all cache management in timing.
Disabled/interval-one paths attach no wrappers and must preserve native output/RNG.
