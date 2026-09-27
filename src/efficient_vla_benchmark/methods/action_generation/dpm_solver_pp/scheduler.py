"""DPM-Solver++ equations supplied by the pinned public scheduler runtime."""

from diffusers import DPMSolverMultistepScheduler


def make_scheduler(betas, order):
    return DPMSolverMultistepScheduler(
        num_train_timesteps=len(betas),
        trained_betas=betas,
        solver_order=order,
        prediction_type="epsilon",
        thresholding=False,
        lower_order_final=True,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        final_sigmas_type="zero",
    )
