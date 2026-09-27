# BEAST

Use [the released processor](https://huggingface.co/zhouhongyi/beast) at
`ec3cb260baaa780d52785e3b64fdb2fb7690c1a9`: closed-form B-spline coefficients,
calibrated coefficient bounds and 256-bin discretization. The default uses ten
basis functions, degree four, all seven raw command dimensions and no boundary
constraints. Bounds are accumulated only from the frozen calibration split;
every reconstruction call uses `update_bounds=False` and verifies immutability.

Five/ten/twenty basis presets expose rate–distortion trade-offs without gradient
training. The optional native zero-order gripper is explicit, not silently enabled.
The default constant-gripper smoothing is included in raw reconstruction error.
No original-action initial-position constraint or future-state side channel is used.
Full fitting, host/device transfers, integer extraction and reconstruction are timed.
