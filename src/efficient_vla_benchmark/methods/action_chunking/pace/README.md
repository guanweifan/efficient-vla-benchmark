# PACE

[Phase-Aware Chunk Execution](https://arxiv.org/html/2606.00537v1), Section 3
and the smoothed-Cartesian ablation: detect prominent low-speed valleys, accept
them against a task-level demonstration-calibrated threshold, then execute to the
earliest accepted boundary or the maximum real prefix. No model training.

This independent single-arm LIBERO port supports OFT's eight and CogACT's sixteen
native predictions through one controller. Both already output Cartesian deltas:
their XYZ displacement norm is the speed profile (up to the constant controller
scale/time interval). Do NOT difference these deltas again. Rotation and gripper
commands are not TCP translation speed. The low-speed kth command marks execution
through k; no padded actions or changed prediction length is introduced.

The paper leaves the smoothing kernel, numerical prominence implementation,
calibration sampling and delta-action indexing unspecified. This port explicitly
uses a centered three-point boxcar with edge padding, SciPy topographic prominence
on negative speed, and full-training-trajectory prominence distributions.
SciPy spacing selects candidate minima before threshold filtering. The default
spacing is two controls for the short native chunks, not the paper's ten for H50.
The threshold is the paper's default fifth percentile; stricter percentile presets
test longer execution. These are disclosed implementation conventions, not claimed
author-code parity. The paper's joint-space default and bimanual experiments are
not reproduced.

Calibration reads only action/task/frame columns from the LeRobot-v2 conversion
of modified LIBERO training demonstrations. Seeded sampling selects ten complete
episodes per task; all forty tasks must have real valleys. No evaluation episodes,
outcomes or per-task Dev-tuned parameters are used. Freeze distributions once;
one global percentile selects all task thresholds. Changing smoothing/spacing
requires a separately compatible training artifact. Dataset/model paths are CLI
inputs. PyArrow is calibration-only; inference needs NumPy/SciPy.

```bash
python -m efficient_vla_benchmark.methods.action_chunking.pace.calibrate \
  --dataset-path "$TRAINING_DATA" --output "$CALIBRATION_DIR"
evb evaluate --config configs/experiments/cogact/pace.yaml \
  --model-path "$MODEL_DIR" --artifact-path "$CALIBRATION_DIR" \
  --libero-root "$LIBERO_ROOT" --output-dir "$RUN_DIR"
```

The original H8 policy remains the comparator. OFT cannot execute more than eight
predictions, so this method may shorten but cannot extend its native prefix.
Neither success improvement nor acceleration is guaranteed. Report full selection
cost, actual queries and horizons, calibration identity, and measured outcomes.
Close/reset clear instruction state; model outputs and RNG remain native.
