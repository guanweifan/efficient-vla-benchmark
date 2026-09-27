"""UniPC equations supplied by the pinned public scheduler runtime."""

from diffusers import UniPCMultistepScheduler


def make_scheduler(betas, order):
    return UniPCMultistepScheduler(
        num_train_timesteps=len(betas),
        trained_betas=betas,
        solver_order=order,
        prediction_type="epsilon",
        thresholding=False,
        lower_order_final=True,
        predict_x0=True,
        solver_type="bh2",
        final_sigmas_type="zero",
    )
