"""DEIS equations supplied by the pinned public scheduler runtime."""

from diffusers import DEISMultistepScheduler


class StableDEISMultistepScheduler(DEISMultistepScheduler):
    """The identical first-order exponential integral, including its clean limit."""

    def deis_first_order_update(self, model_output, *args, sample=None, **kwargs):
        if args or kwargs or sample is None:
            raise ValueError("Only the pinned internal scheduler interface is supported")
        rho_t, rho_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(rho_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(rho_s)
        return (alpha_t / alpha_s) * sample - (alpha_t * sigma_s / alpha_s - sigma_t) * model_output


def make_scheduler(betas, order):
    return StableDEISMultistepScheduler(
        num_train_timesteps=len(betas),
        trained_betas=betas,
        solver_order=order,
        prediction_type="epsilon",
        thresholding=False,
        lower_order_final=True,
        algorithm_type="deis",
        solver_type="logrho",
    )
